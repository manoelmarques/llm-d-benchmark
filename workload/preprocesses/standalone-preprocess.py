#!/usr/bin/env python3

"""Serialize tensorizer file, move HF files to OS page cache"""

import logging
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import psutil


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def kill_process(proc: psutil.Process):
    """kills a process"""
    for child in proc.children(recursive=True):
        if child.is_running():
            child.kill()
            logger.info("child process %d terminated", child.pid)

    if proc.is_running():
        proc.kill()
        logger.info("main process %d terminated", proc.pid)


def vllm_health(
    process: multiprocessing.Process | subprocess.Popen,
    health_counter: list[float],
    health_counter_lock: threading.Lock,
    end_event: threading.Event,
):
    """makes sure vllm process is not stuck"""

    max_health_wait = 15 * 60

    proc = psutil.Process(process.pid)
    while not end_event.is_set() and proc.is_running():
        time.sleep(0.5)

        start = 0.0
        with health_counter_lock:
            start = health_counter[0]

        elapsed = time.perf_counter() - start
        if elapsed > max_health_wait:
            # if vllm hasn't responded
            logger.info(
                "vLLM process is stuck for more than %.2f secs, aborting ...", elapsed
            )
            kill_process(proc)
            return


class PipeTee:
    """sends stdout to pipe"""

    def __init__(self, pipe):
        self.pipe = pipe
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        """writes data"""
        self.stdout.write(data)
        self.pipe.send(data)

    def flush(self):
        """flushes data"""
        self.stdout.flush()

    def __del__(self):
        sys.stdout = self.stdout
        self.stdout = None


def serialize(model: str, tensorizer_uri: str, conn) -> None:
    """serializes a model to disk"""

    _ = PipeTee(conn)

    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.tensorizer import (
        TensorizerConfig,
        tensorize_vllm_model,
    )

    engine_args = EngineArgs(model=model)
    tensorizer_config = TensorizerConfig(tensorizer_uri=tensorizer_uri)

    tensorize_vllm_model(engine_args, tensorizer_config)


def serialize_model(model: str, tensorizer_uri: str) -> None:
    """process to serialize a model to disk"""

    parent_conn, child_conn = multiprocessing.Pipe()

    process = multiprocessing.Process(
        target=serialize,
        args=(model, tensorizer_uri, child_conn),
    )
    process.start()

    # Close the write end of the pipe in the parent process
    child_conn.close()

    health_counter = [time.perf_counter()]
    health_counter_lock = threading.Lock()
    end_health_event = threading.Event()

    health_thread = threading.Thread(
        target=vllm_health,
        args=(process, health_counter, health_counter_lock, end_health_event),
        daemon=True,
    )
    health_thread.start()

    while process.is_alive() or parent_conn.poll():
        if parent_conn.poll():
            try:
                _ = parent_conn.recv()
                # restart health counter
                with health_counter_lock:
                    health_counter[0] = time.perf_counter()
            except EOFError:
                break  # Exit loop when pipe is closed

    # Wait for the child process to finish
    process.join()
    parent_conn.close()
    end_health_event.set()  # end health check event
    health_thread.join()

    if process.exitcode is not None and process.exitcode != 0:
        raise RuntimeError(f"Serialize process exited with code '{process.exitcode}'")


def minimal_ext_cmd(cmd: list[str] | str, shell: bool) -> tuple[str, str, int]:
    """runs simple external command"""

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
        ) as proc:
            stdout, stderr = proc.communicate()
            return (
                stdout.strip().decode("ascii"),
                stderr.strip().decode("ascii"),
                proc.returncode,
            )
    except Exception as e:
        return ("", str(e), 2)


def copy_hf_cached(
    model: str, dest_hf_home: str, model_path_queue: multiprocessing.Queue
) -> None:
    """copies model cached files to new location"""

    # internal import to capture the latest HF env.variables
    import huggingface_hub.constants
    from huggingface_hub import scan_cache_dir

    src_cache = huggingface_hub.constants.HF_HUB_CACHE
    logger.info("src cache %s", src_cache)
    hf_home = huggingface_hub.constants.HF_HOME
    logger.info("hf_home %s", hf_home)
    logger.info("dest_hf_home %s", dest_hf_home)
    end_path = src_cache.removeprefix(hf_home).removeprefix("/")
    dest_cache = pathlib.Path(dest_hf_home).joinpath(end_path)
    logger.info("dest cache %s", dest_cache)

    # if cache directory doesn't exist just return
    if not os.path.isdir(src_cache):
        return

    hf_cache_info = scan_cache_dir()
    repo_path = None
    for repo in hf_cache_info.repos:
        if repo.repo_type != "model":
            continue

        if model == repo.repo_id:
            repo_path = repo.repo_path
            break

    if repo_path is not None:
        dest = pathlib.Path(dest_cache).joinpath(repo_path.name)
        logger.info("copy from repo path %s to dest path %s", repo_path, dest)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            repo_path,
            dest,
            symlinks=True,
            ignore_dangling_symlinks=True,
        )
        model_path_queue.put(dest)


def copy_hf_cached_model(model: str, dest_hf_home: str) -> list[pathlib.Path]:
    """process to copy model cached files to new location"""

    model_path_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=copy_hf_cached,
        args=(model, dest_hf_home, model_path_queue),
    )
    process.start()

    # Wait for the child process to finish
    process.join()
    if process.exitcode is not None and process.exitcode != 0:
        raise RuntimeError(
            f"Copy HF Cache process exited with code '{process.exitcode}'"
        )

    model_path = None if model_path_queue.empty() else model_path_queue.get()

    files: list[pathlib.Path] = []
    if model_path is not None:
        for dirpath, _, filenames in os.walk(model_path):
            parent = pathlib.Path(dirpath)
            for filename in filenames:
                files.append(parent.joinpath(filename))

    return files


def scan_hf_cache(model: str) -> list[pathlib.Path]:
    """returns model cached files"""

    # internal import to capture the latest HF env.variables
    import huggingface_hub.constants
    from huggingface_hub import scan_cache_dir

    files: list[pathlib.Path] = []
    # if cache directory doesn't exist just return no files
    hf_cache = huggingface_hub.constants.HF_HUB_CACHE
    if not os.path.isdir(hf_cache):
        return files

    hf_cache_info = scan_cache_dir()
    repo_path = None
    for repo in hf_cache_info.repos:
        if repo.repo_type != "model":
            continue

        if model == repo.repo_id:
            repo_path = repo.repo_path
            break

    if repo_path is not None:
        for dirpath, _, filenames in os.walk(repo_path):
            parent = pathlib.Path(dirpath)
            for filename in filenames:
                files.append(parent.joinpath(filename))

    return files


def run_vmtouch(touch: bool, cached_files: list[pathlib.Path]):
    """write list to temp file that will be deleted after context manager ends"""
    with tempfile.NamedTemporaryFile(
        prefix="vllm_vmtouch_", suffix=".txt", delete_on_close=False
    ) as tmp:
        with open(tmp.name, "w", encoding="utf8") as f:
            for cached_file in cached_files:
                f.write(f"{cached_file.resolve()}\n")

        cmd = ["vmtouch"]
        if touch:
            cmd.append("-t")
        cmd.extend(["-b", tmp.name])
        try:
            logger.info("start command: %s", " ".join(cmd))
            output, err, rc = minimal_ext_cmd(cmd, False)
            if output != "":
                logger.info(output)
            if rc != 0:
                raise RuntimeError(
                    f"Command '{' '.join(cmd)}' exited with code '{rc}': '{err}'"
                )
        finally:
            logger.info("end command: %s", " ".join(cmd))


def get_env_variables(keys: list[str]) -> list[str]:
    """get environment variables"""

    logger.info("Environment variables:")

    env_vars = os.environ

    envs = []
    missing_envs = []
    for key in keys:
        value = env_vars.get(key)
        if value is None:
            missing_envs.append(key)
        else:
            envs.append(value)
            logger.info("  '%s': '%s'", key, value)

    if len(missing_envs) > 0:
        raise RuntimeError(f"Env. variables not found: {','.join(missing_envs)}.")
    return envs


def preprocess_run() -> str:
    """preprocess function"""

    envs = get_env_variables(
        [
            "LLMDBENCH_VLLM_STANDALONE_VLLM_LOAD_FORMAT",
            "LLMDBENCH_VLLM_STANDALONE_MODEL",
            "LLMDBENCH_VLLM_STANDALONE_IN_PAGE_CACHE",
            "LLMDBENCH_VLLM_TENSORIZER_URI",
            "LLMDBENCH_VLLM_NEW_HF_HOME",
        ]
    )

    load_format = envs[0].strip().lower()
    model = envs[1].strip()
    in_page_cache = envs[2].strip().lower()
    tensorizer_uri = envs[3]
    new_hf_home = envs[4]

    if load_format == "tensorizer":
        # first serialize model in order to run tokenizer library later
        try:
            logger.info(
                "Start model %s serialization for tokenizer library to %s",
                model,
                tensorizer_uri,
            )
            serialize_model(model, tensorizer_uri)
            logger.info("Model %s serialized to %s", model, tensorizer_uri)
        finally:
            logger.info("End model %s serialization", model)

    cached_files = []
    if in_page_cache in ["yes", "true"]:
        # load model cached files into OS page cache
        if load_format == "tensorizer":
            cached_files = [pathlib.Path(tensorizer_uri)]
        else:
            cached_files = scan_hf_cache(model)

        run_vmtouch(True, cached_files)
    elif in_page_cache in ["no", "false"]:
        # copy cached model to new cache location
        # since they are new files, they won't be in OS page cache
        # no need to deal with tensorizer serialized file as it is a new file
        cached_files = copy_hf_cached_model(model, new_hf_home)

    print_cached_files(cached_files)


def print_cached_files(cached_files: list[pathlib.Path]) -> None:
    """print cached files"""
    logger.info("Cached files:")
    rows = [["File"]]
    for cached_file in cached_files:
        rows.append(
            [
                str(cached_file),
            ]
        )
    print_table(rows)


def print_table(table: list[list[str]]) -> None:
    """prints a matrix with header and rows in table format"""
    if len(table) == 0:
        return

    longest_cols = [len(max(col, key=len)) + 3 for col in zip(*table)]
    row_format = "".join(
        ["{:>" + str(longest_col) + "}" for longest_col in longest_cols]
    )
    # print header
    header = table[0]
    logger.info(row_format.format(*header))
    row_underline = ["-" * longest_col for longest_col in longest_cols]
    # print underline
    logger.info(row_format.format(*row_underline))
    # print rows
    for row in table[1:]:
        logger.info(row_format.format(*row))


if __name__ == "__main__":
    try:
        logger.info("Start preprocess run")
        preprocess_run()
    except Exception as e:
        logger.error("Error running preprocess: %s", str(e))
    finally:
        logger.info("End preprocess run")

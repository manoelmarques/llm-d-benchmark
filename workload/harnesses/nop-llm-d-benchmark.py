#!/usr/bin/env python3

"""
Benchmark 'nop' harness
"""

from __future__ import annotations
from datetime import datetime, timezone
import os
import logging
from pathlib import Path
import yaml

from nop_functions import (
    get_env_variables,
    BenchmarkResult,
    convert_result,
    LoadFormat,
    benchmark_nop,
)

from fma_functions import benchmark_fma

from kubernetes import client, config

# Configure logging
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60.0  # time (seconds) to wait for request
MAX_VLLM_WAIT = 15.0 * 60.0  # time (seconds) to wait for vllm to respond


def setup_logger(directory: str):
    """Configure the root logger"""

    Path(directory).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{directory}/stdout.log"),
            logging.StreamHandler(),
        ],
    )


def main():
    """main entry point"""

    start_time = datetime.now().astimezone(timezone.utc).timestamp()

    write_log_per_process = False

    logger.info("Environment variables:")
    envs = get_env_variables(
        [
            "LLMDBENCH_CONTROL_WORK_DIR",
            "LLMDBENCH_DEPLOY_METHODS",
        ]
    )

    requests_dir = envs["LLMDBENCH_CONTROL_WORK_DIR"]
    deploy_methods = envs["LLMDBENCH_DEPLOY_METHODS"]

    setup_logger(requests_dir)
    for key, value in envs.items():
        logger.info("  '%s': '%s'", key, value)

    keys = [
        "LLMDBENCH_HARNESS_NAMESPACE",
        "LLMDBENCH_HARNESS_STACK_ENDPOINT_URL",
        "LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT",
        "LLMDBENCH_HARNESS_STACK_ENDPOINT_LAUNCHER_URL",
        "LLMDBENCH_HARNESS_STACK_ENDPOINT_LAUNCHER_VLLM_URL",
    ]
    if "standalone" in deploy_methods:
        keys.extend(
            [
                "LLMDBENCH_VLLM_STANDALONE_LAUNCHER",
                "LLMDBENCH_VLLM_STANDALONE_LAUNCHER_VLLM_PORT",
            ]
        )
    if "fma" in deploy_methods:
        keys.extend(["LLMDBENCH_FMA_LAUNCHER_CONFIG_PORT", "LLMDBENCH_FMA_ITERATIONS"])

    envs = get_env_variables(keys)
    for key, value in envs.items():
        logger.info("  '%s': '%s'", key, value)

    namespace = envs["LLMDBENCH_HARNESS_NAMESPACE"]
    endpoint_url = envs["LLMDBENCH_HARNESS_STACK_ENDPOINT_URL"]
    load_format = LoadFormat.loadformat_from_value(
        envs["LLMDBENCH_VLLM_COMMON_VLLM_LOAD_FORMAT"]
    )
    launcher = (
        envs["LLMDBENCH_VLLM_STANDALONE_LAUNCHER"].strip().lower() == "true"
        if "standalone" in deploy_methods
        else False
    )
    endpoint_launcher_url = envs["LLMDBENCH_HARNESS_STACK_ENDPOINT_LAUNCHER_URL"]
    endpoint_launcher_vllm_url = envs[
        "LLMDBENCH_HARNESS_STACK_ENDPOINT_LAUNCHER_VLLM_URL"
    ]
    launcher_vllm_port = (
        envs["LLMDBENCH_VLLM_STANDALONE_LAUNCHER_VLLM_PORT"]
        if "standalone" in deploy_methods
        else None
    )
    fma_launcher_port = (
        envs["LLMDBENCH_FMA_LAUNCHER_CONFIG_PORT"] if "fma" in deploy_methods else 0
    )

    fma_iterations = int(
        envs["LLMDBENCH_FMA_ITERATIONS"] if "fma" in deploy_methods else 0
    )

    # Load Kubernetes configuration
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    api = client.CustomObjectsApi()

    benchmark_result = BenchmarkResult()
    benchmark_result.scenario.deploy_methods = deploy_methods
    if "fma" in deploy_methods:
        try:
            logger.info("Benchmark FMA launcher start...")
            benchmark_fma(
                v1,
                api,
                apps_v1,
                namespace,
                endpoint_url,
                fma_launcher_port,
                benchmark_result,
                load_format,
                requests_dir,
                fma_iterations,
                REQUEST_TIMEOUT,
                MAX_VLLM_WAIT,
                write_log_per_process,
            )
        except Exception:
            logger.exception("error on benchmark FMA launcher")
        finally:
            logger.info("Benchmark FMA launcher end")
    else:
        benchmark_nop(
            v1,
            apps_v1,
            namespace,
            endpoint_url,
            endpoint_launcher_url,
            endpoint_launcher_vllm_url,
            launcher_vllm_port,
            launcher,
            benchmark_result,
            load_format,
            requests_dir,
            REQUEST_TIMEOUT,
            MAX_VLLM_WAIT,
            write_log_per_process,
        )

    stop_time = datetime.now().astimezone(timezone.utc).timestamp()
    benchmark_result.time.start = start_time
    benchmark_result.time.stop = stop_time

    # write results yaml file
    result_filepath = os.path.join(requests_dir, "result.yaml")
    with open(result_filepath, "w", encoding="utf-8", newline="") as file:
        yaml.dump(benchmark_result.dump(), file, indent=2, sort_keys=False)
        logger.info("result yaml file saved to path: %s", result_filepath)

    benchmark_report_filepath = os.path.join(requests_dir, "benchmark_report")
    os.makedirs(benchmark_report_filepath, exist_ok=True)
    benchmark_report_filepath = os.path.join(benchmark_report_filepath, "result.yaml")
    convert_result(result_filepath, benchmark_report_filepath, start_time, stop_time)


if __name__ == "__main__":
    try:
        logger.info("Starting 'nop' harness run")
        main()
    except Exception:
        logger.exception("Error running 'nop' harness")
    finally:
        logger.info("End 'nop' harness run")

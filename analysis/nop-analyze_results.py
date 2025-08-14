#!/usr/bin/env python3

"""
Startup logs benchmark
"""

import io
import os
import logging
from typing import Any
import pandas as pd
import yaml

from schema import BenchmarkReport

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def get_formatted_output(column: str, df: pd.DataFrame) -> str:
    """get formatted output"""
    max_len = df[column].astype(str).str.len().max()
    formatters = {column: lambda x: f"{x:<{max_len}}"}
    df_string = df.to_string(formatters=formatters, index=False)

    lines = df_string.split("\n")
    separator = "-" * len(lines[0])

    # Insert the separator after the header line
    lines.insert(1, separator)

    return f"{'\n'.join(lines)}\n"


def create_categories_dataframe(
    categories: list[dict[str, Any]],
    level: int,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """create categories dataframe"""

    blank_string = "  " * level if level > 0 else ""
    total = 0.0
    for category in categories:
        elapsed = category.get("elapsed", 0)
        total += elapsed
        elapsed_str = f"{elapsed:.3f}" if elapsed != 0 else ""
        data = {
            "Category": [category["title"]],
            "Elapsed(secs)": [elapsed_str],
        }
        data = pd.DataFrame(data)
        data.iloc[0, 0] = blank_string + data.iloc[0, 0]
        df = pd.concat([df, data])

        children = category.get("categories")
        if children is not None:
            df = create_categories_dataframe(children, level + 1, df)

    df_total = pd.DataFrame(
        {
            "Category": [blank_string + "Total"],
            "Elapsed(secs)": [total],
        }
    )

    # Append the total row to the DataFrame
    return pd.concat([df, df_total])


def write_benchmark_reports(file: io.TextIOWrapper, benchmark_report: BenchmarkReport):
    """write benchmark reports to file"""

    scenario_model_name = benchmark_report.scenario.model.name
    scenario_metadata = benchmark_report.scenario.metadata
    metrics_metadata = benchmark_report.metrics.metadata
    transfer_rate = 0.0
    if metrics_metadata["load_time"] > 0:
        transfer_rate = metrics_metadata["size"] / metrics_metadata["load_time"]
    data = {
        "Time": [metrics_metadata["time"]],
        "vLLM Version": [scenario_metadata["vllm_version"]],
        "Sleep/Wake": [str(scenario_metadata["sleep_mode"])],
        "Model": [scenario_model_name],
        "Load Format": [scenario_metadata["load_format"]],
        "Elapsed(secs)": [f"{metrics_metadata['load_time']:.2f}"],
        "Rate(GB/s)": [f"{transfer_rate:.2f}"],
        "Sleep(secs)": [f"{metrics_metadata['sleep']:.2f}"],
        "Freed GPU(GiB)": [f"{metrics_metadata['gpu_freed']:.2f}"],
        "In Use GPU(GiB)": [f"{metrics_metadata['gpu_in_use']:.2f}"],
        "Wake(secs)": [f"{metrics_metadata['wake']:.2f}"],
    }
    data_frame = pd.DataFrame(data)
    file.write(get_formatted_output("Time", data_frame))

    categories = metrics_metadata.get("categories")
    if categories is None:
        return

    file.write("\n\n\n\n")
    data_frame = create_categories_dataframe(categories, 0, pd.DataFrame())
    file.write(get_formatted_output("Category", data_frame))


def main():
    """main entry point"""

    envs = get_env_variables(
        [
            "LLMDBENCH_CONTROL_WORK_DIR",
        ]
    )

    control_work_dir = envs[0]
    requests_dir = control_work_dir

    # read possible existent universal yaml file
    benchmark_report_filepath = os.path.join(
        requests_dir, "benchmark_report", "result.yaml"
    )
    if not os.path.isfile(benchmark_report_filepath):
        logger.info(
            "no benchmark reports file found on path: %s", benchmark_report_filepath
        )
        return

    benchmark_report = None
    with open(benchmark_report_filepath, "r", encoding="UTF-8") as file:
        benchmark_dict = yaml.safe_load(file)
        benchmark_report = BenchmarkReport(**benchmark_dict)

    analysis_dir = os.path.join(requests_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # write reports analysis file
    reports_filepath = os.path.join(analysis_dir, "result.txt")
    with open(reports_filepath, "w", encoding="utf-8") as file:
        write_benchmark_reports(file, benchmark_report)
        logger.info("analysis report file saved to path: %s", reports_filepath)


if __name__ == "__main__":
    try:
        logger.info("Starting analysis run")
        main()
    except Exception:
        logger.exception("Error running analysis")
    finally:
        logger.info("End analysis run")

import datetime
import os

import numpy as np
import pandas as pd

ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
ts_name = f"{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}"

BASE_PATH = os.path.abspath(".")
CSV_RESULT_DIR = os.path.join(BASE_PATH, "csv_results")
OUTPUT_DIR = os.path.join(BASE_PATH, "analysis_results", ts_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASK_SET = [
    "AntMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "DKittyMorphology-Exact-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    "HopperController-Exact-v0",
    "gtopx_data_2_1" "gtopx_data_3_1" "gtopx_data_4_1" "gtopx_data_6_1",
]

METHOD_SET = [
    "baseline_omnipred_24m",
    "baseline_embed_regress_proj_t5",
    "baseline_embed_regress_t5_m_cat_from_scratch",
    "Expert_GA",
    "Expert_Grad",
]


def read_csv_results(csv_path):
    df = pd.read_csv(csv_path)
    return df


def analyze_scores():
    # Get list of all CSV files
    score_files = {"50th": [], "100th": []}

    for file in os.listdir(CSV_RESULT_DIR):
        if file.endswith(".csv"):
            if "normalized-score" in file:
                if "50th" in file:
                    score_files["50th"].append(file)
                elif "100th" in file:
                    score_files["100th"].append(file)

    results = {}
    for percentile in ["50th", "100th"]:
        results[percentile] = []
        for file in score_files[percentile]:
            df = read_csv_results(os.path.join(CSV_RESULT_DIR, file))
            results[percentile].append(df)
    # print(results['50th'][0])
    # assert False

    # Calculate statistics
    statistics = {}
    for percentile in ["50th", "100th"]:
        statistics[percentile] = {}
        for percentile in results[percentile]:
            if results[percentile][percentile]:
                combined_df = pd.concat(results[percentile][percentile])
                mean_df = combined_df.groupby("task").mean()
                std_df = combined_df.groupby("task").std()
                statistics[score_type][percentile] = {"mean": mean_df, "std": std_df}

    # Save results
    for score_type in statistics:
        for percentile in statistics[score_type]:
            output_file = f"{score_type}_{percentile}_statistics.csv"
            result_df = pd.DataFrame()
            if statistics[score_type][percentile]:
                mean_df = statistics[score_type][percentile]["mean"]
                std_df = statistics[score_type][percentile]["std"]
                result_df = pd.DataFrame(
                    {"mean": mean_df.iloc[:, 0], "std": std_df.iloc[:, 0]}
                )
                result_df.to_csv(os.path.join(OUTPUT_DIR, output_file))


if __name__ == "__main__":
    analyze_scores()

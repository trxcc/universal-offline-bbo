import datetime
import os

import numpy as np
import pandas as pd

ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
ts_name = f"{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}"


def read_csv_results(csv_path):
    return pd.read_csv(csv_path)


BASE_PATH = os.path.abspath(".")
CSV_RESULT_DIR = os.path.join(BASE_PATH, "csv_results")
OUTPUT_DIR = os.path.join(BASE_PATH, "analysis_results", ts_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original task and method sets
TASK_SET = [
    "AntMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "DKittyMorphology-Exact-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    "HopperController-Exact-v0",
    "gtopx_data_2_1",
    "gtopx_data_3_1",
    "gtopx_data_4_1",
    "gtopx_data_6_1",
]

METHOD_SET = [
    "D(best)",
    "baseline_omnipred_24m",
    "baseline_embed_regress_proj_t5",
    "baseline_embed_regress_t5_m_cat_from_scratch",
    "Expert_GA",
    "Expert_Grad",
]

D_best_dict = {
    "AntMorphology-Exact-v0": 165.32648,
    "Superconductor-RandomForest-v0": 74.0,
    "DKittyMorphology-Exact-v0": 199.36252,
    "TFBind8-Exact-v0": 0.43929616,
    "TFBind10-Exact-v0": 0.005328223,
    "HopperController-Exact-v0": 1361.6106,
    "gtopx_data_2_1": -195.58598182625983,
    "gtopx_data_3_1": -151.18979217783007,
    "gtopx_data_4_1": -215.71556768924154,
    "gtopx_data_6_1": -112.59916211889862,
}


def format_value(mean, std):
    """Format mean and std to the desired string format"""
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    return f"{mean:.3f} ± {std:.3f}"


def create_performance_table(data_frames):
    """Create performance table with tasks as rows and methods as columns"""
    # Initialize DataFrame with tasks as rows and methods as columns
    results_df = pd.DataFrame(index=TASK_SET + ["avg_rank"], columns=METHOD_SET)
    results_df.index.name = "Task"

    # Fill D(best) values
    for task in TASK_SET:
        results_df.at[task, "D(best)"] = format_value(D_best_dict[task], 0)

    combined_df = pd.concat(data_frames, ignore_index=True)

    # Process each task and method
    for task in TASK_SET:
        for method in METHOD_SET[1:]:  # Skip D(best)
            task_data = combined_df[combined_df["task"] == task]
            if not task_data.empty and method in task_data.columns:
                mean = task_data[method].mean()
                std = task_data[method].std()
                results_df.at[task, method] = format_value(mean, std)

    # Calculate average rank with std
    method_ranks = {method: [] for method in METHOD_SET}

    for df in data_frames:
        for task in TASK_SET:
            task_data = df[df["task"] == task]
            if not task_data.empty:
                task_values = []
                task_values.append(("D(best)", D_best_dict[task]))

                for method in METHOD_SET[1:]:
                    if method in task_data.columns:
                        value = task_data[method].iloc[0]
                        task_values.append((method, value))

                sorted_methods = sorted(task_values, key=lambda x: x[1], reverse=True)
                for rank, (method, _) in enumerate(sorted_methods, 1):
                    method_ranks[method].append(rank)

    # Calculate mean and std of ranks for each method
    rank_means = {}
    for method in METHOD_SET:
        if method_ranks[method]:
            rank_mean = np.mean(method_ranks[method])
            rank_std = np.std(method_ranks[method])
            rank_means[method] = rank_mean
            results_df.at["avg_rank", method] = format_value(rank_mean, rank_std)
        else:
            results_df.at["avg_rank", method] = "N/A"

    best_method = min(rank_means.items(), key=lambda x: x[1])[0]
    results_df.at["avg_rank", best_method] = (
        f"\\textbf{{{results_df.at['avg_rank', best_method]}}}"
    )

    return results_df


def analyze_scores():
    # Get list of all CSV files
    score_files = {"50th": [], "100th": []}

    # Collect relevant CSV files
    for file in os.listdir(CSV_RESULT_DIR):
        if file.endswith(".csv"):
            if "normalized-score" not in file:
                if "50th" in file:
                    score_files["50th"].append(file)
                elif "100th" in file:
                    score_files["100th"].append(file)

    # Process each percentile
    for percentile in ["50th", "100th"]:
        data_frames = []
        for file in score_files[percentile]:
            df = read_csv_results(os.path.join(CSV_RESULT_DIR, file))
            df = df[df["task"].isin(TASK_SET)]
            if not df.empty:
                data_frames.append(df)

        if data_frames:
            # Create performance table
            perf_table = create_performance_table(data_frames)

            # Save results
            output_file = f"performance_table_{percentile}.csv"
            perf_table.to_csv(os.path.join(OUTPUT_DIR, output_file))

            # Create formatted version with highlighted best performance
            formatted_table = highlight_best_performance(perf_table.copy())
            formatted_file = f"performance_table_{percentile}_formatted.csv"
            formatted_table.to_csv(os.path.join(OUTPUT_DIR, formatted_file))


def highlight_best_performance(df):
    """Highlight best performance in each row"""

    def extract_mean(x):
        if isinstance(x, str) and "±" in x:
            return float(x.split("±")[0].strip())
        return np.nan

    # Create a DataFrame with just the means
    means_df = df.loc[df.index.drop("avg_rank")].applymap(extract_mean)

    # For each row, bold the highest mean value
    for idx in df.index:
        if idx != "avg_rank":  # Skip avg_rank row
            row_means = means_df.loc[idx]
            max_mean = row_means.max()
            for col in df.columns:
                if means_df.loc[idx, col] == max_mean:
                    value = df.at[idx, col]
                    df.at[idx, col] = f"\\textbf{{{value}}}"

    return df


if __name__ == "__main__":
    analyze_scores()

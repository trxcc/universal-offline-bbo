# import os
# import pandas as pd
# import numpy as np
# import datetime

# ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
# ts_name = f'{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'

# BASE_PATH = os.path.abspath(".")
# CSV_RESULT_DIR = os.path.join(BASE_PATH, "csv_results")
# OUTPUT_DIR = os.path.join(BASE_PATH, "analysis_results", ts_name)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# TASK_SET = [
#     "AntMorphology-Exact-v0",
#     "Superconductor-RandomForest-v0",
#     "DKittyMorphology-Exact-v0",
#     "TFBind8-Exact-v0",
#     "TFBind10-Exact-v0",
#     "HopperController-Exact-v0",
#     "gtopx_data_2_1",
#     "gtopx_data_3_1",
#     "gtopx_data_4_1",
#     "gtopx_data_6_1"
# ]

# METHOD_SET = [
#     "baseline_omnipred_24m",
#     "baseline_embed_regress_proj_t5",
#     "baseline_embed_regress_t5_m_cat_from_scratch",
#     "Expert_GA",
#     "Expert_Grad",
# ]

# def format_value(mean, std):
#     """Format mean and std to the desired string format"""
#     if pd.isna(mean) or pd.isna(std):
#         return "N/A"
#     return f"{mean:.2f} ± {std:.2f}"

# def read_csv_results(csv_path):
#     return pd.read_csv(csv_path)

# def create_performance_table(data_frames):
#     """Create performance table for specified tasks and methods"""
#     # Initialize DataFrame with tasks as rows and methods as columns
#     results_df = pd.DataFrame(index=TASK_SET, columns=METHOD_SET)

#     # Combine all data frames
#     combined_df = pd.concat(data_frames, ignore_index=True)

#     # Process each task and method
#     for task in TASK_SET:
#         task_data = combined_df[combined_df['task'] == task]
#         if not task_data.empty:
#             for method in METHOD_SET:
#                 if method in task_data.columns:
#                     mean = task_data[method].mean()
#                     std = task_data[method].std()
#                     results_df.at[task, method] = format_value(mean, std)

#     return results_df

# def analyze_scores():
#     # Get list of all CSV files
#     score_files = {
#         '50th': [],
#         '100th': []
#     }

#     # Collect relevant CSV files
#     for file in os.listdir(CSV_RESULT_DIR):
#         if file.endswith('.csv'):
#             if 'normalized-score' in file:
#                 if '50th' in file:
#                     score_files['50th'].append(file)
#                 elif '100th' in file:
#                     score_files['100th'].append(file)

#     # Process each percentile
#     for percentile in ['50th', '100th']:
#         data_frames = []
#         for file in score_files[percentile]:
#             df = read_csv_results(os.path.join(CSV_RESULT_DIR, file))
#             # Filter for only the tasks we're interested in
#             df = df[df['task'].isin(TASK_SET)]
#             if not df.empty:
#                 data_frames.append(df)

#         if data_frames:
#             # Create performance table
#             perf_table = create_performance_table(data_frames)

#             # Save results
#             output_file = f"performance_table_{percentile}.csv"
#             perf_table.to_csv(os.path.join(OUTPUT_DIR, output_file))

#             # Create formatted version with highlighted best performance
#             formatted_table = highlight_best_performance(perf_table.copy())
#             formatted_file = f"performance_table_{percentile}_formatted.csv"
#             formatted_table.to_csv(os.path.join(OUTPUT_DIR, formatted_file))

# def highlight_best_performance(df):
#     """Highlight best performance in each column"""
#     def extract_mean(x):
#         if isinstance(x, str) and '±' in x:
#             return float(x.split('±')[0].strip())
#         return np.nan

#     # Create a DataFrame with just the means
#     means_df = df.applymap(extract_mean)

#     # For each column, bold the highest mean value
#     for col in df.columns:
#         max_mean = means_df[col].max()
#         for idx in df.index:
#             if means_df.at[idx, col] == max_mean:
#                 value = df.at[idx, col]
#                 df.at[idx, col] = f"\\textbf{{{value}}}"

#     return df

import datetime

# if __name__ == "__main__":
#     analyze_scores()
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
    "baseline_omnipred_24m",
    "baseline_embed_regress_proj_t5",
    "baseline_embed_regress_t5_m_cat_from_scratch",
    "Expert_GA",
    "Expert_Grad",
]


def format_value(mean, std):
    """Format mean and std to the desired string format"""
    if pd.isna(mean) or pd.isna(std):
        return "N/A"
    return f"{mean:.2f} ± {std:.2f}"


def create_performance_table(data_frames):
    """Create performance table for specified tasks and methods"""
    # Initialize DataFrame with methods as rows and tasks as columns
    # This is inverted from the original to match the image format
    results_df = pd.DataFrame(index=METHOD_SET, columns=["D(best)"] + TASK_SET)

    # Combine all data frames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Leave D(best) column empty for manual filling
    results_df["D(best)"] = ""

    # Process each method and task
    for method in METHOD_SET:
        for task in TASK_SET:
            task_data = combined_df[combined_df["task"] == task]
            if not task_data.empty and method in task_data.columns:
                mean = task_data[method].mean()
                std = task_data[method].std()
                results_df.at[method, task] = format_value(mean, std)

    return results_df


def analyze_scores():
    # Get list of all CSV files
    score_files = {"50th": [], "100th": []}

    # Collect relevant CSV files
    for file in os.listdir(CSV_RESULT_DIR):
        if file.endswith(".csv"):
            if "normalized-score" in file:
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
    """Highlight best performance in each column"""

    def extract_mean(x):
        if isinstance(x, str) and "±" in x:
            return float(x.split("±")[0].strip())
        return np.nan

    # Create a DataFrame with just the means
    means_df = df.applymap(extract_mean)

    # For each column, bold the highest mean value
    for col in df.columns:
        if col != "D(best)":  # Skip D(best) column
            max_mean = means_df[col].max()
            for idx in df.index:
                if means_df.at[idx, col] == max_mean:
                    value = df.at[idx, col]
                    df.at[idx, col] = f"\\textbf{{{value}}}"

    return df


if __name__ == "__main__":
    analyze_scores()

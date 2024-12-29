import os
import glob
import argparse
import itertools
import pandas as pd
from enum import Enum
from loguru import logger
from utils import CheckType

profiler_dir = "./profiler"
output_dir = "./analyze_results"
utils_ratio = 0.85
split_gemm_ratio = 0.9


def find_matching_rows(df, target_indices):
    result = []
    
    for idx in target_indices:
        row = df.iloc[idx]
        batch_target = row['batch']
        time_target = row['kernel_time']
        

        previous_rows = df.iloc[:idx]
        
        best_time_sum = float('inf')
        best_pair = None
        

        for row1, row2 in itertools.combinations(previous_rows.itertuples(index=False), 2):
            if row1.batch + row2.batch == batch_target:
                time_sum = row1.kernel_time + row2.kernel_time
                if time_sum < best_time_sum:
                    best_time_sum = time_sum
                    best_pair = (row1, row2)
        

        if best_pair and best_time_sum < time_target * split_gemm_ratio:
            row1, row2 = best_pair
            result.append({
                'batch': batch_target,
                'kernel_time': time_target,
                'row1_batch': row1.batch,
                'row1_time': row1.kernel_time,
                'row2_batch': row2.batch,
                'row2_time': row2.kernel_time,
            })
    

    return pd.DataFrame(result)

def analyze_torchgemm_shape_utils(df: pd.DataFrame, output_file_base_name: str, check_split_gemm: bool):
    outlier_rows = []
    outlier_row_nums = []
    for i in range(1, len(df)):
        prev_data = df.iloc[:i]["utils"]
        current = df.iloc[i]["utils"]

        max_d = prev_data.max()
        max_d_index = prev_data.idxmax()
        if max_d * utils_ratio > current:
            outlier_rows.append(df.iloc[i])
            outlier_row_nums.append(i)
            current_batch = df.iloc[i]["batch"]
            max_batch = df.iloc[max_d_index]["batch"]
            logger.info(f"batch {current_batch} has utils {current}, which is lower than {utils_ratio} of batch {max_batch}, with utils {max_d}")
            
    outiler_df = pd.DataFrame(outlier_rows)
    low_utils_output_file = os.path.join(output_dir, "low_utils_" + output_file_base_name + ".csv")

    logger.info(f"Of all the {df.shape[0]} batches, found {outiler_df.shape[0]} batches which has lower utils than "
                f"smaller gemm, full results is stored in {low_utils_output_file}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outiler_df_without_kernel = outiler_df.drop('kernel_name', axis=1)

    outiler_df_without_kernel.to_csv(low_utils_output_file, index=False)
    
    if check_split_gemm:
        split_gemm_output_file = os.path.join(output_dir, "split_gemm_" + output_file_base_name + ".csv")
        gemm_df = find_matching_rows(df, outlier_row_nums)
        gemm_df.to_csv(split_gemm_output_file, index=False)
        logger.info(f"Of all the {outiler_df.shape[0]} batches, found {gemm_df.shape[0]} batches"
                    f"which can be optimized by split_gemm, and the result is saved at {split_gemm_output_file}")


def analyze_shape_utils(csv_file: str, check_split_gemm: bool):
    assert "torchgemm" in csv_file
    df = pd.read_csv(csv_file)
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_file_base_name = file_name
    analyze_torchgemm_shape_utils(df, output_file_base_name, check_split_gemm)


def parse_arguments():
    parser = argparse.ArgumentParser(description="analyze the profiler results and find performance issue")    
    parser.add_argument(
        "--profiler", 
        type=str, 
        required=True,
        help="the profiler file to read"
    )

    parser.add_argument(
        "--check_split_gemm", 
        action='store_true',
        help="whether to perform split_gemm check"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    profiler_path = args.profiler
    check_split_gemm = args.check_split_gemm
    analyze_shape_utils(profiler_path, check_split_gemm)


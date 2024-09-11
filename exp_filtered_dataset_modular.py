import os
import pandas as pd
import numpy as np
import shutil
import re
from typing import List, Union, Callable, Dict
import scipy.stats as stats

# Constants for directory paths
BASE_DIR = "/Users/andrew/Thesis/smart-image-augmentation"
REAL_DIR = os.path.join(BASE_DIR, "pascal", "real")
AUG_DIR = os.path.join(BASE_DIR, "pascal", "aug-pascal-original", "pascal-0-1")
FILTRATION_CSV_PATH = os.path.join(BASE_DIR, "results", "filtration-results", "pascal-0-1-results.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "pascal", "filtered-pascal-0-1")

# Configuration
SCORE_COLUMNS = ['class_representation', 'visual_fidelity', 'structural_integrity', 'diversity', 'beneficial_variations']
CREATIVE_SCORE_PAIRS = [(0, 3), (2, 4)]  # Indices of columns to be multiplied for creative score

def calculate_score(row, creative: bool = False):
    if creative:
        return sum(row[SCORE_COLUMNS[i]] * row[SCORE_COLUMNS[j]] for i, j in CREATIVE_SCORE_PAIRS)
    else:
        return sum(row[col] for col in SCORE_COLUMNS) / len(SCORE_COLUMNS)

def filter_images(csv_path: str, aug_dir: str, output_dir: str, strategies: List[dict]):
    df = pd.read_csv(csv_path)
    seed, example_per_class = extract_info_from_csv_name(csv_path)

    for strategy in strategies:
        creative = strategy.get('creative', False)
        
        if strategy['type'] not in ['zscore_top_n', 'percentile_top_n']:
            df['score'] = df.apply(lambda row: calculate_score(row, creative), axis=1)

        filtered_df = apply_strategy(df, strategy)
        strategy_name = strategy['name']
        strategy_value = strategy.get('value', '')
        if strategy_value == '':
            strategy_value = 'NA'
        strategy_output_dir = create_output_dir(output_dir, seed, example_per_class, strategy_name, strategy_value)
        copy_filtered_images(filtered_df, aug_dir, strategy_output_dir)
        print_strategy_results(strategy_name, strategy_value, strategy_output_dir, filtered_df)

def apply_strategy(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    strategy_type = strategy['type']
    if strategy_type == 'top_n':
        return apply_top_n_strategy(df, strategy['value'], strategy.get('group_by_class', True))
    elif strategy_type == 'percentile':
        return apply_percentile_strategy(df, strategy['value'], strategy.get('group_by_class', True))
    elif strategy_type == 'zscore':
        return apply_zscore_strategy(df, strategy['threshold'], strategy.get('group_by_class', True))
    elif strategy_type == 'percentile_by_columns':
        return apply_percentile_by_columns_strategy(df, strategy['value'], strategy.get('group_by_class', True))
    elif strategy_type == 'zscore_top_n':
        return apply_zscore_top_n_strategy(df, strategy['zscore_threshold'], strategy['top_n_value'], strategy.get('group_by_class', True), strategy.get('creative', False))
    elif strategy_type == 'percentile_top_n':
        return apply_percentile_top_n_strategy(df, strategy['percentile_value'], strategy['top_n_value'], strategy.get('group_by_class', True), strategy.get('creative', False))
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def apply_top_n_strategy(df: pd.DataFrame, n: float, group_by_class: bool) -> pd.DataFrame:
    if group_by_class:
        return df.groupby('class_name').apply(lambda x: x.nlargest(max(1, int(len(x) * n)), 'score')).reset_index(drop=True)
    else:
        return df.nlargest(max(1, int(len(df) * n)), 'score')

def apply_percentile_strategy(df: pd.DataFrame, p: float, group_by_class: bool) -> pd.DataFrame:
    if group_by_class:
        return df[df['score'] > df.groupby('class_name')['score'].transform(lambda x: x.quantile(p))]
    else:
        return df[df['score'] > df['score'].quantile(p)]

def apply_zscore_strategy(df: pd.DataFrame, threshold: float, group_by_class: bool) -> pd.DataFrame:
    if group_by_class:
        return df.groupby('class_name').apply(lambda x: x[~(np.abs(stats.zscore(x[SCORE_COLUMNS])) > threshold).any(axis=1)]).reset_index(drop=True)
    else:
        return df[~(np.abs(stats.zscore(df[SCORE_COLUMNS])) > threshold).any(axis=1)]

def apply_percentile_by_columns_strategy(df: pd.DataFrame, percentile: float, group_by_class: bool) -> pd.DataFrame:
    if group_by_class:
        return df.groupby('class_name').apply(lambda x: x[np.all(x[SCORE_COLUMNS] > x[SCORE_COLUMNS].quantile(percentile), axis=1)]).reset_index(drop=True)
    else:
        return df[np.all(df[SCORE_COLUMNS] > df[SCORE_COLUMNS].quantile(percentile), axis=1)]

def apply_zscore_top_n_strategy(df: pd.DataFrame, zscore_threshold: float, top_n_value: float, group_by_class: bool, creative: bool) -> pd.DataFrame:
    # First, apply Z-score filtering
    zscore_filtered = df[~(np.abs(stats.zscore(df[SCORE_COLUMNS])) > zscore_threshold).any(axis=1)]
    
    # Then, calculate scores and apply top-n
    zscore_filtered['score'] = zscore_filtered.apply(lambda row: calculate_score(row, creative), axis=1)
    return apply_top_n_strategy(zscore_filtered, top_n_value, group_by_class)

def apply_percentile_top_n_strategy(df: pd.DataFrame, percentile_value: float, top_n_value: float, group_by_class: bool, creative: bool) -> pd.DataFrame:
    # First, apply percentile filtering
    percentile_filtered = df[np.all(df[SCORE_COLUMNS] > df[SCORE_COLUMNS].quantile(percentile_value), axis=1)]
    
    # Then, calculate scores and apply top-n
    percentile_filtered['score'] = percentile_filtered.apply(lambda row: calculate_score(row, creative), axis=1)
    return apply_top_n_strategy(percentile_filtered, top_n_value, group_by_class)

def create_output_dir(base_dir: str, seed: str, example_per_class: str, strategy: str, value: Union[float, str]) -> str:
    if strategy == 'combined':
        strategy_dir = f"filtered-pascal-{seed}-{example_per_class}-combined-{value}".replace(".", "_")
    else:
        if isinstance(value, float):
            strategy_dir = f"filtered-pascal-{seed}-{example_per_class}-{strategy}-{value:.2f}".replace(".", "_")
        else:
            strategy_dir = f"filtered-pascal-{seed}-{example_per_class}-{strategy}-{value}".replace(".", "_")
    output_dir = os.path.join(base_dir, strategy_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_filtered_images(df: pd.DataFrame, src_dir: str, dst_dir: str):
    for _, row in df.iterrows():
        src_path = os.path.join(src_dir, row['filename'])
        dst_path = os.path.join(dst_dir, row['filename'])
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: File not found: {src_path}")

def create_real_images_folder(csv_path: str, real_images_dir: str, output_dir: str):
    df = pd.read_csv(csv_path)
    seed, example_per_class = extract_info_from_csv_name(csv_path)
    real_folder_name = f"real-{seed}-{example_per_class}"
    real_output_dir = os.path.join(output_dir, real_folder_name)
    
    if os.path.exists(real_output_dir):
        print(f"Folder {real_output_dir} already exists. Skipping creation.")
        return
    
    os.makedirs(real_output_dir, exist_ok=True)
    
    train_dir = os.path.join(real_output_dir, 'train')
    val_dir = os.path.join(real_output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for class_name in df['class_name'].unique():
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        class_df = df[df['class_name'] == class_name]
        real_filenames = class_df['real_filename'].unique()
        
        # Copy images to train folder
        for filename in real_filenames[:int(example_per_class)]:
            src_path = os.path.join(real_images_dir, 'train', class_name, filename)
            dst_path = os.path.join(train_class_dir, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Real file not found: {src_path}")
        
        # Copy all images from real folder to val folder
        src_val_dir = os.path.join(real_images_dir, 'val', class_name)
        if os.path.exists(src_val_dir):
            for filename in os.listdir(src_val_dir):
                src_path = os.path.join(src_val_dir, filename)
                dst_path = os.path.join(val_class_dir, filename)
                shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Validation folder not found: {src_val_dir}")
    
    print(f"Real images folder created: {real_output_dir}")

def extract_info_from_csv_name(csv_path: str) -> tuple:
    filename = os.path.basename(csv_path)
    match = re.search(r'pascal-(\d+)-(\d+)-results', filename)
    if match:
        return match.group(1), match.group(2)
    else:
        raise ValueError(f"Unable to extract seed and example_per_class from filename: {filename}")

def print_strategy_results(strategy_name: str, strategy_value: Union[float, str], output_dir: str, filtered_df: pd.DataFrame):
    print(f"Strategy: {strategy_name} (value: {strategy_value})")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {len(filtered_df)}")
    print("Images per class:")
    print(filtered_df['class_name'].value_counts())
    print("\n")

def main():
    # Create real images folder
    create_real_images_folder(FILTRATION_CSV_PATH, REAL_DIR, OUTPUT_DIR)
    
    # Define strategies
    strategies = [
        # Top-n strategies
        {'type': 'top_n', 'value': 0.9, 'group_by_class': True, 'name': 'top_n_class', 'creative': False},
        {'type': 'top_n', 'value': 0.9, 'group_by_class': False, 'name': 'top_n_overall', 'creative': False},
        {'type': 'top_n', 'value': 0.9, 'group_by_class': True, 'name': 'top_n_class_creative', 'creative': True},
        {'type': 'top_n', 'value': 0.9, 'group_by_class': False, 'name': 'top_n_overall_creative', 'creative': True},

        # Percentile strategies
        {'type': 'percentile', 'value': 0.1, 'group_by_class': True, 'name': 'percentile_class', 'creative': False},
        {'type': 'percentile', 'value': 0.1, 'group_by_class': False, 'name': 'percentile_overall', 'creative': False},
        {'type': 'percentile', 'value': 0.1, 'group_by_class': True, 'name': 'percentile_class_creative', 'creative': True},
        {'type': 'percentile', 'value': 0.1, 'group_by_class': False, 'name': 'percentile_overall_creative', 'creative': True},

        # Z-score strategy
        {'type': 'zscore', 'threshold': 2, 'group_by_class': True, 'name': 'zscore_class'},
        {'type': 'zscore', 'threshold': 2, 'group_by_class': False, 'name': 'zscore_overall'},

        # Percentile by columns strategy
        {'type': 'percentile_by_columns', 'value': 0.1, 'group_by_class': True, 'name': 'percentile_columns_class'},
        {'type': 'percentile_by_columns', 'value': 0.1, 'group_by_class': False, 'name': 'percentile_columns_overall'},

        # Combined Z-score + top-n strategy
        {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': True, 'name': 'zscore_top_n_class', 'creative': False},
        {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'zscore_top_n_overall', 'creative': False},
        {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': True, 'name': 'zscore_top_n_class_creative', 'creative': True},
        {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'zscore_top_n_overall_creative', 'creative': True},
        
        # Combined Column-level Percentile + top-n strategy
        {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': True, 'name': 'percentile_top_n_class', 'creative': False},
        {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'percentile_top_n_overall', 'creative': False},
        {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': True, 'name': 'percentile_top_n_class_creative', 'creative': True},
        {'type': 'percentile_top_n', 'percentile_value': 0.1, 'top_n_value': 0.9, 'group_by_class': False, 'name': 'percentile_top_n_overall_creative', 'creative': True},
    ]
    
    # Filter images
    filter_images(FILTRATION_CSV_PATH, AUG_DIR, OUTPUT_DIR, strategies)

if __name__ == "__main__":
    main()
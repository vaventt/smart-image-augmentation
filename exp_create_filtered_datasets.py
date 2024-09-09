import os
import pandas as pd
import shutil
import re
from typing import List, Union

def filter_images(csv_path: str, aug_dir: str, output_dir: str, top_n: List[float] = None, percentile: List[float] = None):
    df = pd.read_csv(csv_path)
    seed, example_per_class = extract_info_from_csv_name(csv_path)

    if top_n and percentile and len(top_n) == len(percentile):
        for p, n in zip(percentile, top_n):
            filtered_df = apply_combined_strategy(df, p, n)
            strategy_output_dir = create_output_dir(output_dir, seed, example_per_class, 'combined', f"{p:.2f}-{n:.2f}")
            copy_filtered_images(filtered_df, aug_dir, strategy_output_dir)
            print(f"Combined filtered dataset (percentile {p:.2f}, top_n {n:.2f}) saved to {strategy_output_dir}")
            print(f"Total images: {len(filtered_df)}")
            print("Images per class:")
            print(filtered_df['class_name'].value_counts())
            print("\n")
    else:
        strategies = []
        if top_n:
            strategies.extend([('top_n', value) for value in top_n])
        if percentile:
            strategies.extend([('percentile', value) for value in percentile])
        
        if not strategies:
            raise ValueError("At least one filtering strategy (top_n or percentile) must be provided.")
        
        for strategy, value in strategies:
            strategy_output_dir = create_output_dir(output_dir, seed, example_per_class, strategy, value)
            
            if strategy == 'top_n':
                filtered_df = apply_top_n_strategy(df, value)
            elif strategy == 'percentile':
                filtered_df = apply_percentile_strategy(df, value)
            
            copy_filtered_images(filtered_df, aug_dir, strategy_output_dir)
            print(f"Filtered dataset for {strategy}_{value:.2f} saved to {strategy_output_dir}")
            print(f"Total images: {len(filtered_df)}")
            print("Images per class:")
            print(filtered_df['class_name'].value_counts())
            print("\n")

def apply_top_n_strategy(df: pd.DataFrame, n: float) -> pd.DataFrame:
    return df.groupby('class_name').apply(lambda x: x.nlargest(max(1, int(len(x) * n)), 'overall_score')).reset_index(drop=True)

def apply_percentile_strategy(df: pd.DataFrame, p: float) -> pd.DataFrame:
    return df[df['overall_score'] > df.groupby('class_name')['overall_score'].transform(lambda x: x.quantile(p))]

def apply_combined_strategy(df: pd.DataFrame, p: float, n: float) -> pd.DataFrame:
    percentile_filtered = apply_percentile_strategy(df, p)
    return apply_top_n_strategy(percentile_filtered, n)

def create_output_dir(base_dir: str, seed: str, example_per_class: str, strategy: str, value: Union[float, str]) -> str:
    if strategy == 'combined':
        strategy_dir = f"filtered-pascal-{seed}-{example_per_class}-combined-{value}".replace(".", "_")
    else:
        strategy_dir = f"filtered-pascal-{seed}-{example_per_class}-{strategy}-{value:.2f}".replace(".", "_")
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
    _, example_per_class = extract_info_from_csv_name(csv_path)
    real_folder_name = f"real-{example_per_class}"
    real_output_dir = os.path.join(output_dir, real_folder_name)
    
    if os.path.exists(real_output_dir):
        print(f"Folder {real_output_dir} already exists. Skipping creation.")
        return
    
    os.makedirs(real_output_dir, exist_ok=True)
    
    for class_name in df['class_name'].unique():
        class_dir = os.path.join(real_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        class_df = df[df['class_name'] == class_name]
        real_filenames = class_df['real_filename'].unique()
        
        for filename in real_filenames:
            src_path = os.path.join(real_images_dir, 'train', class_name, filename)
            dst_path = os.path.join(class_dir, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Real file not found: {src_path}")
    
    print(f"Real images folder created: {real_output_dir}")

def extract_info_from_csv_name(csv_path: str) -> tuple:
    filename = os.path.basename(csv_path)
    match = re.search(r'pascal-(\d+)-(\d+)_results', filename)
    if match:
        return match.group(1), match.group(2)
    else:
        raise ValueError(f"Unable to extract seed and example_per_class from filename: {filename}")

def main():
    base_dir = "/Users/andrew/Thesis/smart-image-augmentation/pascal"
    real_images_dir = os.path.join(base_dir, "real")
    
    # Placeholder lists for filtering strategies
    top_n_values = [0.9, 0.8, 0.7]
    percentile_values = [0.1, 0.2, 0.3]
    
    # CSV file and augmented images directory
    csv_path = os.path.join(base_dir, "filtration_results", "pascal-0-1_results.csv")
    aug_dir = os.path.join(base_dir, "aug-pascal-original", "pascal-0-1")
    
    # Create real images folder
    create_real_images_folder(csv_path, real_images_dir, base_dir)
    
    # Filter images
    filter_images(csv_path, aug_dir, base_dir, top_n_values, percentile_values)

if __name__ == "__main__":
    main()
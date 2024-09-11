import os
import pandas as pd
import shutil
from typing import List, Union

def filter_images(csv_path: str, aug_dir: str, output_dir: str, top_n: List[float] = None, percentile: List[float] = None):
    """
    Filter images based on GPT-4 analysis results.
    
    :param csv_path: Path to the CSV file with GPT-4 analysis results
    :param aug_dir: Directory containing the augmented images
    :param output_dir: Directory to save filtered images
    :param top_n: List of float values for top_n filtering strategy
    :param percentile: List of float values for percentile filtering strategy
    """
    df = pd.read_csv(csv_path)
    
    strategies = []
    if top_n:
        strategies.extend([('top_n', value) for value in top_n])
    if percentile:
        strategies.extend([('percentile', value) for value in percentile])
    
    if not strategies:
        raise ValueError("At least one filtering strategy (top_n or percentile) must be provided.")
    
    for strategy, value in strategies:
        strategy_output_dir = create_output_dir(output_dir, strategy, value)
        
        if strategy == 'top_n':
            filtered_df = apply_top_n_strategy(df, value)
        elif strategy == 'percentile':
            filtered_df = apply_percentile_strategy(df, value)
        
        copy_filtered_images(filtered_df, aug_dir, strategy_output_dir)
        print(f"Filtered dataset for {strategy}_{value} saved to {strategy_output_dir}")
        print(f"Total images: {len(filtered_df)}")
        print("Images per class:")
        print(filtered_df['class_name'].value_counts())
        print("\n")

def apply_top_n_strategy(df: pd.DataFrame, n: float) -> pd.DataFrame:
    return df.groupby('class_name').apply(lambda x: x.nlargest(max(1, int(len(x) * n)), 'overall_score')).reset_index(drop=True)

def apply_percentile_strategy(df: pd.DataFrame, p: float) -> pd.DataFrame:
    return df[df['overall_score'] > df.groupby('class_name')['overall_score'].transform(lambda x: x.quantile(p))]

def create_output_dir(base_dir: str, strategy: str, value: float) -> str:
    aug_folder = os.path.basename(os.path.dirname(base_dir))
    strategy_dir = f"filtered-{aug_folder}-{strategy}_{value:.2f}".replace(".", "_")
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
    """
    Create a folder with real images based on the CSV file.
    
    :param csv_path: Path to the CSV file with GPT-4 analysis results
    :param real_images_dir: Directory containing the real images
    :param output_dir: Directory to save the real images
    """
    df = pd.read_csv(csv_path)
    example_per_class = len(df[df['class_name'] == df['class_name'].iloc[0]]) // 10  # Assuming 10 augmented images per real image
    real_folder_name = f"real-{example_per_class}"
    real_output_dir = os.path.join(output_dir, real_folder_name)
    
    if os.path.exists(real_output_dir):
        print(f"Folder {real_output_dir} already exists. Skipping creation.")
        return
    
    os.makedirs(real_output_dir, exist_ok=True)
    
    real_filenames = df['real_filename'].unique()
    for filename in real_filenames:
        class_name = df[df['real_filename'] == filename]['class_name'].iloc[0]
        src_path = os.path.join(real_images_dir, 'train', class_name, filename)
        dst_path = os.path.join(real_output_dir, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Real file not found: {src_path}")
    
    print(f"Real images folder created: {real_output_dir}")

def main():
    base_dir = "/Users/andrew/Thesis/smart-image-augmentation/pascal"
    real_images_dir = os.path.join(base_dir, "real")
    
    # Placeholder lists for filtering strategies
    top_n_values = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]
    percentile_values = None
    
    # CSV file and augmented images directory
    csv_path = os.path.join(base_dir, "filtration_results", "pascal-0-1_results.csv")
    aug_dir = os.path.join(base_dir, "aug-pascal-original", "pascal-0-1")
    
    # Create real images folder
    create_real_images_folder(csv_path, real_images_dir, base_dir)
    
    # Filter images
    filter_images(csv_path, aug_dir, base_dir, top_n_values, percentile_values)

if __name__ == "__main__":
    main()
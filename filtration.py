import os
import re
import base64
import pandas as pd
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Constants for directory paths
BASE_DIR = "/Users/andrew/Thesis/smart-image-augmentation"
REAL_DIR = os.path.join(BASE_DIR, "pascal", "real", "train")
AUG_BASE_DIR = os.path.join(BASE_DIR, "pascal", "aug-pascal-original")
MAPPING_BASE_DIR = os.path.join(BASE_DIR, "pascal", "map-images")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "filtration-results")

# Constants for pricing
INPUT_TOKEN_PRICE_PER_MILLION = 5
OUTPUT_TOKEN_PRICE_PER_MILLION = 15

# Pascal classes in order
PASCAL_CLASSES = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep',
                  'sofa', 'train', 'television']

def fix_json_string(json_string):
    # Fix misplaced quotation marks
    json_string = re.sub(r'([^\\])"([^:,\[\]{}]+)":', r'\1"\2":', json_string)
    
    # Fix missing commas between objects
    json_string = re.sub(r'}(\s*){', r'},\1{', json_string)
    
    return json_string

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def analyze_images_with_gpt(image_paths, class_name):
    encoded_images = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}", "detail": "low"}}
        for path in image_paths if encode_image(path) is not None
    ]
    
    if not encoded_images:
        print(f"No valid images to analyze for class {class_name}")
        return None, 0, 0
    
    prompt = f"""You are evaluating augmented images for an image classification task. The first image of class {class_name} is real and serves as reference.
    For each of the remaining 10 augmented images, please evaluate them based on the following criteria, providing a score between 0.0 and 1.0 for each:
    
    Index: from 1 to 10 in order of the augmented images

    1. Class Representation: How well does the image maintain the core semantic content, key features, and distinguishability of the original class? (0 = completely irrelevant or indistinguishable, 1 = perfectly relevant, feature-complete, and highly distinguishable)
    2. Visual Fidelity: Is the image clear, well-formed, free from artifacts, natural-looking, and in an appropriate context for the given class? (0 = poor quality, unrealistic, or out of context, 1 = excellent quality, realistic, and in perfect context)
    3. Structural Integrity: How well does the image maintain the overall structure and proportions expected for the class? (0 = severely distorted structure, 1 = perfect structural integrity)
    4. Diversity: How different is the image from the reference while still maintaining class identity? (0 = identical, 1 = maximum beneficial diversity)
    5. Beneficial Variations: How effectively does the image introduce meaningful changes in pose, viewpoint, lighting, or color that could aid in generalization? (0 = no variation, 1 = optimal variation)

    Provide your response as a list of JSON objects, one for each non-reference augmented image, in this format:
    [
    {{
    "index": 1,
    "class_representation": 0.0,
    "visual_fidelity": 0.0,
    "structural_integrity": 0.0,
    "diversity": 0.0,
    "beneficial_variations": 0.0,
    "explanation": "Brief explanation of how changes affect classification and quality"
    }}
    ]
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *encoded_images
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.0
        )

        content = response.choices[0].message.content.strip('`').strip()
        if content.startswith('json'):
            content = content[4:]
        
        # Attempt to fix and parse JSON
        try:
            fixed_content = fix_json_string(content)
            parsed_content = json.loads(fixed_content)
            return parsed_content, response.usage.prompt_tokens, response.usage.completion_tokens
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error for class {class_name}")
            print(f"Original content: {content}")
            print(f"Fixed content: {fixed_content}")
            print(f"JSON Error: {str(json_error)}")
            
            # Attempt to salvage partial results
            partial_results = []
            for item in re.finditer(r'{[^}]+}', fixed_content):
                try:
                    partial_item = json.loads(item.group())
                    partial_results.append(partial_item)
                except json.JSONDecodeError:
                    continue
            
            if partial_results:
                print(f"Salvaged {len(partial_results)} partial results")
                return partial_results, response.usage.prompt_tokens, response.usage.completion_tokens
            else:
                return None, response.usage.prompt_tokens, response.usage.completion_tokens

    except Exception as e:
        print(f"An error occurred during GPT analysis for class {class_name}: {e}")
        return None, 0, 0

def calculate_price(input_tokens, output_tokens):
    input_price = (input_tokens / 1_000_000) * INPUT_TOKEN_PRICE_PER_MILLION
    output_price = (output_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE_PER_MILLION
    return input_price + output_price

def process_image_group(real_path, aug_paths, class_name):
    start_time = time.time()

    print(f"Augmented images: {[os.path.basename(path) for path in aug_paths]}")
    
    # Check if the real image exists
    if not os.path.exists(real_path):
        print(f"Error: Real image not found - {real_path}")
        return None, 0, 0, 0

    all_paths = [real_path] + aug_paths
    results, input_tokens, output_tokens = analyze_images_with_gpt(all_paths, class_name)
    
    df_data = []
    if results:
        for i, result in enumerate(results):
            if i < len(aug_paths):
                result['filename'] = os.path.basename(aug_paths[i])
                result['class_name'] = class_name
                df_data.append(result)
    
    # Add default values for missing or incomplete results
    while len(df_data) < len(aug_paths):
        i = len(df_data)
        df_data.append({
            'index': i + 1,
            'class_representation': 0.5,
            'visual_fidelity': 0.5,
            'structural_integrity': 0.5,
            'diversity': 0.5,
            'beneficial_variations': 0.5,
            'explanation': 'Default values due to incomplete GPT response',
            'filename': os.path.basename(aug_paths[i]),
            'class_name': class_name
        })
    
    df = pd.DataFrame(df_data)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return df, execution_time, input_tokens, output_tokens

def process_folder(real_dir, aug_dir, mapping_file, output_dir):
    all_results = []
    total_execution_time = 0
    total_input_tokens = 0
    total_output_tokens = 0

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'{os.path.basename(aug_dir)}-log.txt')
    
    # Read the mapping file
    mapping_df = pd.read_csv(mapping_file)
    print(f"Mapping file columns: {mapping_df.columns}")
    
    # Group the mapping dataframe by real filename and class
    grouped = mapping_df.groupby(['filename_real', 'class'])
    
    total_groups = len(grouped)
    processed_groups = 0
    
    with open(log_file, 'w') as log:
        for class_name in PASCAL_CLASSES:
            class_groups = [group for name, group in grouped if name[1] == class_name]
            for group in class_groups:
                real_filename = group['filename_real'].iloc[0]
                aug_filenames = group['filename_aug'].tolist()
                
                real_path = os.path.join(real_dir, class_name, real_filename)
                aug_paths = [os.path.join(aug_dir, aug_filename) for aug_filename in aug_filenames]
                
                print(f"Processing group: {class_name} - {real_filename} ({processed_groups + 1}/{total_groups})")
                
                class_results, execution_time, input_tokens, output_tokens = process_image_group(real_path, aug_paths, class_name)
                
                if class_results is not None:
                    class_results['real_filename'] = real_filename
                    all_results.append(class_results)
                    total_execution_time += execution_time
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    price = calculate_price(input_tokens, output_tokens)
                    log_message = f"Processed group: {class_name} - {real_filename}, Time: {execution_time:.2f}s, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Price: ${price:.4f}"
                else:
                    log_message = f"Skipped group: {class_name} - {real_filename} due to missing real image"
                
                log.write(log_message + '\n')
                print(log_message)
                
                processed_groups += 1
                
                # Add a delay to avoid rate limiting
                time.sleep(10)  # Wait for 10 seconds between requests

        total_price = calculate_price(total_input_tokens, total_output_tokens)
        summary = f"\nTotal execution time: {total_execution_time:.2f}s\nTotal input tokens: {total_input_tokens}\nTotal output tokens: {total_output_tokens}\nTotal price: ${total_price:.4f}"
        log.write(summary)
        print(summary)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        
        csv_path = os.path.join(output_dir, f'{os.path.basename(aug_dir)}-results.csv')
        final_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print(f"No results were obtained for {aug_dir}")

def get_matching_mapping_file(folder, mapping_dir):
    folder_parts = folder.split('-')
    if len(folder_parts) != 3:
        raise ValueError(f"Invalid folder name format: {folder}")
    
    seed, examples_per_class = folder_parts[1], folder_parts[2]
    pattern = f"images-map-{seed}-{examples_per_class}.csv"
    
    for file in os.listdir(mapping_dir):
        if file == pattern:
            return os.path.join(mapping_dir, file)
    
    raise FileNotFoundError(f"No matching mapping file found for folder {folder}")

def main():
    # Get all folders in AUG_BASE_DIR that match the pattern "pascal-X-Y"
    folder_pattern = re.compile(r'pascal-\d+-\d+')
    folders_to_process = [folder for folder in os.listdir(AUG_BASE_DIR) if folder_pattern.match(folder)]

    for folder in folders_to_process:
        aug_dir = os.path.join(AUG_BASE_DIR, folder)
        try:
            mapping_file = get_matching_mapping_file(folder, MAPPING_BASE_DIR)
            print(f"Processing folder: {folder}")
            print(f"Mapping file: {mapping_file}")
            process_folder(REAL_DIR, aug_dir, mapping_file, OUTPUT_DIR)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing folder {folder}: {str(e)}")

    print("All folders processed.")

if __name__ == "__main__":
    main()
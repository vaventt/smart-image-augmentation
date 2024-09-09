import os
import base64
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_images_with_gpt(image_paths, class_name, num_real=5):
    encoded_images = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}}
        for path in image_paths
    ]
    
    prompt = f"""Analyze these images of class {class_name}. The first {num_real} images are real and serve as reference.
    For each of the remaining augmented images, compare them to the real ones and provide float scores from 0 to 1 for:
    Index: from 1 to 10 in order of the augmented images
    a) Quality: Overall visual fidelity and clarity compared to the real ones.
    b) Realism: How well it matches real-world expectations set by the reference images.
    c) Relevance: How well it represents a {class_name} compared to the reference images.
    d) Detail Preservation: Retention of important class-specific features.
    
    Respond ONLY with a list of JSON objects, one for each non-reference image, in this format:
    [
      {{"index": 1, "quality": 0.0, "realism": 0.0, "relevance": 0.0, "detail_preservation": 0.0, "explanation": "Brief explanation."}}
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
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip('`').strip()
        if content.startswith('json'):
            content = content[4:]
        
        try:
            parsed_content = json.loads(content)
            return parsed_content[:10] if len(parsed_content) > 10 else parsed_content, response.usage.total_tokens
        except json.JSONDecodeError:
            print(f"JSON parsing error for class {class_name}")
            return None, response.usage.total_tokens

    except Exception as e:
        print(f"An error occurred during GPT analysis for class {class_name}: {e}")
        return None, 0

def process_images(real_dir, aug_dir, class_name, num_real=5):
    start_time = time.time()
    class_names = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                   'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 
                   'sofa', 'train', 'television']
    
    class_index = class_names.index(class_name)
    
    real_images = random.sample([f for f in os.listdir(real_dir) if f.endswith('.jpg')], num_real)
    real_paths = [os.path.join(real_dir, img) for img in real_images]
    
    aug_images = [f for f in os.listdir(aug_dir) if f.startswith(f'aug-{class_index}-')]
    aug_paths = [os.path.join(aug_dir, img) for img in aug_images]
    
    all_paths = real_paths + aug_paths
    print(len(all_paths))
    
    results, tokens_used = analyze_images_with_gpt(all_paths, class_name, num_real)
    
    if results:
        df_data = []
        for i, filename in enumerate(aug_paths):
            if i < len(results):
                result = results[i]
                result['filename'] = os.path.basename(filename)
                result['class_name'] = class_name  # Add this line
                df_data.append(result)
            else:
                df_data.append({
                    'filename': os.path.basename(filename),
                    'index': -1,
                    'quality': 0.5,
                    'realism': 0.5,
                    'relevance': 0.5,
                    'detail_preservation': 0.5,
                    'explanation': 'No analysis provided by GPT',
                    'class_name': class_name  # Add this line
                })
        
        df = pd.DataFrame(df_data)
        df['overall_score'] = df[['quality', 'realism', 'relevance', 'detail_preservation']].mean(axis=1)
        df = df.sort_values('overall_score', ascending=False)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return df, execution_time, tokens_used
    
    return None, time.time() - start_time, 0

def process_folder(real_dir, aug_dir, output_dir):
    all_results = []
    total_execution_time = 0
    total_tokens_used = 0
    class_names = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                   'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 
                   'sofa', 'train', 'television']

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'{os.path.basename(aug_dir)}_log.txt')
    
    with open(log_file, 'w') as log:
        for class_name in class_names:
            try:
                class_results, execution_time, tokens_used = process_images(os.path.join(real_dir, class_name), aug_dir, class_name)
                if class_results is not None:
                    all_results.append(class_results)
                    total_execution_time += execution_time
                    total_tokens_used += tokens_used
                    log_message = f"Processed class: {class_name}, Time: {execution_time:.2f}s, Tokens: {tokens_used}"
                    log.write(log_message + '\n')
                    print(log_message)
                
                # Add a delay to avoid rate limiting
            #   time.sleep(60)  # Wait for 20 seconds between requests
            except Exception as exc:
                error_message = f'{class_name} generated an exception: {exc}'
                log.write(error_message + '\n')
                print(error_message)

        summary = f"\nTotal execution time: {total_execution_time:.2f}s\nTotal tokens used: {total_tokens_used}"
        log.write(summary)
        print(summary)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df.sort_values('overall_score', ascending=False)
        
        csv_path = os.path.join(output_dir, f'{os.path.basename(aug_dir)}_results.csv')
        final_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print(f"No results were obtained for {aug_dir}")

# Main execution
if __name__ == "__main__":
    BASE_DIR = "/Users/andrew/Thesis/smart-image-augmentation"
    REAL_DIR = os.path.join(BASE_DIR, "pascal", "real", "train")
    AUG_BASE_DIR = os.path.join(BASE_DIR, "aug-pascal")
    OUTPUT_DIR = os.path.join(BASE_DIR, "filtration_results")

    # List of folders to process
    folders_to_process = [
        "pascal-0-1"
        # Add more folders as needed
    ]

    for folder in folders_to_process:
        aug_dir = os.path.join(AUG_BASE_DIR, folder)
        print(f"Processing folder: {folder}")
        process_folder(REAL_DIR, aug_dir, OUTPUT_DIR)

    print("All folders processed.")
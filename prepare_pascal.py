import os
import numpy as np
from PIL import Image
from collections import defaultdict
import shutil

def transform_pascal_voc_2012(voc_path, output_path):
    segmentation_path = os.path.join(voc_path, 'SegmentationClass')
    images_path = os.path.join(voc_path, 'JPEGImages')
    image_sets_path = os.path.join(voc_path, 'ImageSets', 'Segmentation')
    segmentation_object_path = os.path.join(voc_path, 'SegmentationObject')
    segmentation_class_path = os.path.join(voc_path, 'SegmentationClass')
    
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    
    def load_image_set(filename):
        with open(os.path.join(image_sets_path, filename), 'r') as f:
            return set(line.strip() for line in f)

    train_set = load_image_set('train.txt')
    val_set = load_image_set('val.txt')
    
    print(f"Number of entries in train set: {len(train_set)}")
    print(f"Number of entries in val set: {len(val_set)}")
    
    # Original class names (for reading segmentation masks)
    original_class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
        'sofa', 'train', 'tvmonitor']
    
    # New class names (for saving in output folders)
    PASCAL_CLASSES = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep',
                      'sofa', 'train', 'television']
    
    # Mapping from original to new class names
    class_name_mapping = {orig: new for orig, new in zip(original_class_names[1:], PASCAL_CLASSES)}
    
    def process_image_set(image_set, output_subset):
        count_dict = defaultdict(int)
        skipped_no_mask = 0
        skipped_background = 0
        
        for image_id in image_set:
            seg_path = os.path.join(segmentation_object_path, f"{image_id}.png")
            if not os.path.exists(seg_path):
                skipped_no_mask += 1
                continue
            
            seg_img = np.array(Image.open(seg_path))
            class_img = np.array(Image.open(os.path.join(segmentation_class_path, f"{image_id}.png")))
            
            # Exclude background (0) and boundary/void (255)
            valid_pixels = (seg_img > 0) & (seg_img < 255)
            if not np.any(valid_pixels):
                skipped_background += 1
                continue
            
            # Find the largest object
            objects, counts = np.unique(seg_img[valid_pixels], return_counts=True)
            largest_object = objects[np.argmax(counts)]
            
            # Get the class of the largest object
            largest_class = class_img[seg_img == largest_object][0]
            
            original_class_name = original_class_names[largest_class]
            new_class_name = class_name_mapping[original_class_name]
            
            src_path = os.path.join(images_path, f"{image_id}.jpg")
            dst_dir = os.path.join(output_path, output_subset, new_class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, f"{image_id}.jpg")
            
            shutil.copy(src_path, dst_path)
            count_dict[new_class_name] += 1
        
        print(f"\nProcessed {output_subset} set:")
        print(f"Skipped {skipped_no_mask} images with no segmentation mask")
        print(f"Skipped {skipped_background} images with only background")
        return count_dict
    
    train_count = process_image_set(train_set, 'train')
    val_count = process_image_set(val_set, 'val')
    
    print(f"\nTransformed dataset saved to {output_path}")
    print("\nTraining set summary:")
    for class_name, count in train_count.items():
        print(f"{class_name}: {count} images")
    print(f"Total training images: {sum(train_count.values())}")

    print("\nValidation set summary:")
    for class_name, count in val_count.items():
        print(f"{class_name}: {count} images")
    print(f"Total validation images: {sum(val_count.values())}")

    print(f"\nTotal images processed: {sum(train_count.values()) + sum(val_count.values())}")

# Usage
transform_pascal_voc_2012('/Users/andrew/Thesis/smart-image-augmentation/VOCdevkit/VOC2012', '/Users/andrew/Thesis/smart-image-augmentation/pascal/real')
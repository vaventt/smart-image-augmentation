import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict

class FilteredPASCALDataset(Dataset):
    class_names = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 
                   'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 
                   'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 
                   'sofa', 'train', 'television']
    
    num_classes = len(class_names)

    def __init__(self, real_dir, filtered_dir, filtered_strategy, split='train', 
                 examples_per_class=None, seed=0, synthetic_probability=0.5, 
                 image_size=(256, 256)):
        self.real_dir = real_dir
        self.filtered_dir = filtered_dir
        self.filtered_strategy = filtered_strategy
        self.split = split
        self.examples_per_class = examples_per_class
        self.synthetic_probability = synthetic_probability
        self.image_size = image_size

        self.rng = np.random.default_rng(seed)
        
        self.class_to_images = {cls: [] for cls in self.class_names}
        self.class_to_synthetic = {cls: [] for cls in self.class_names}
        
        self.load_images()
        self.load_synthetic_images()

        self.all_images = []
        self.all_labels = []
        for i, cls in enumerate(self.class_names):
            self.all_images.extend(self.class_to_images[cls])
            self.all_labels.extend([i] * len(self.class_to_images[cls]))

        print(f"Dataset {split}: Loaded {len(self.all_images)} real images and {sum(len(v) for v in self.class_to_synthetic.values())} synthetic images")

        self.transform = self.get_transform(split == 'train')

    def load_images(self):
        for cls in self.class_names:
            class_dir = os.path.join(self.real_dir, self.split, cls)
            if os.path.exists(class_dir):
                images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.jpg')]
                if self.examples_per_class:
                    images = self.rng.choice(images, size=self.examples_per_class, replace=False)
                self.class_to_images[cls] = images

    def load_synthetic_images(self):
        if self.split == 'train':
            filtered_strategy_dir = os.path.join(self.filtered_dir, self.filtered_strategy)
            if os.path.exists(filtered_strategy_dir):
                for i, cls in enumerate(self.class_names):
                    synthetic_images = []
                    for j in range(self.examples_per_class):
                        class_synthetic = [os.path.join(filtered_strategy_dir, img) 
                                           for img in os.listdir(filtered_strategy_dir) 
                                           if img.startswith(f'aug-{i * self.examples_per_class + j}-')]
                        synthetic_images.extend(class_synthetic)
                    self.class_to_synthetic[cls] = synthetic_images

    def get_transform(self, is_train):
        if is_train:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        if self.split == 'train' and np.random.uniform() < self.synthetic_probability:
            cls = self.class_names[self.all_labels[idx]]
            if self.class_to_synthetic[cls]:
                img_path = random.choice(self.class_to_synthetic[cls])
                label = self.all_labels[idx]
            else:
                img_path, label = self.all_images[idx], self.all_labels[idx]
        else:
            img_path, label = self.all_images[idx], self.all_labels[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label
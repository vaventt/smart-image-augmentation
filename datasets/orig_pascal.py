import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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

        self.real_images = []
        self.synthetic_images = []

        np.random.seed(seed)
        random.seed(seed)
        self.load_images()

        print(f"Dataset {split}: Loaded {len(self.real_images)} real images and {len(self.synthetic_images)} synthetic images")

        self.transform = self.get_transform(split == 'train')

    def load_images(self):
        # Load real images
        for cls in self.class_names:
            class_dir = os.path.join(self.real_dir, self.split, cls)
            if os.path.exists(class_dir):
                class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.jpg')]
                self.real_images.extend([(img, self.class_names.index(cls)) for img in class_images[:self.examples_per_class]])

        # Load filtered (synthetic) images
        if self.split == 'train':
            filtered_strategy_dir = os.path.join(self.filtered_dir, self.filtered_strategy)
            if os.path.exists(filtered_strategy_dir):
                for i, cls in enumerate(self.class_names):
                    cls_images = [img for img in os.listdir(filtered_strategy_dir) if img.startswith(f'aug-{i}-')]
                    self.synthetic_images.extend([(os.path.join(filtered_strategy_dir, img), i) for img in cls_images])

    def get_transform(self, is_train):
        if is_train:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15.0),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                     std=[0.5, 0.5, 0.5])
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
        return len(self.real_images) + len(self.synthetic_images)

    def __getitem__(self, idx):
        if self.split == 'train' and np.random.uniform() < self.synthetic_probability and self.synthetic_images:
            img_path, label = random.choice(self.synthetic_images)
        else:
            img_path, label = self.real_images[idx % len(self.real_images)]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label
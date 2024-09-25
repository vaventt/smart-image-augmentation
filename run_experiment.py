import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import trange
import pandas as pd
import random
import numpy as np
from dataset import FilteredPASCALDataset

# Constants
BASE_DIR = "/home/ubuntu/EzLogz/smart-image-augmentation/pascal"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Configuration
EXAMPLES_PER_CLASS = [1,2,4,8,16]
SEEDS = [0]

class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.image_processor = None
        if backbone == "resnet50":
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)
    
    def forward(self, image):
        x = image
        if self.backbone == "resnet50":
            with torch.no_grad():
                x = self.base_model.conv1(x)
                x = self.base_model.bn1(x)
                x = self.base_model.relu(x)
                x = self.base_model.maxpool(x)
                x = self.base_model.layer1(x)
                x = self.base_model.layer2(x)
                x = self.base_model.layer3(x)
                x = self.base_model.layer4(x)
                x = self.base_model.avgpool(x)
                x = torch.flatten(x, 1)
        return self.out(x)

def run_experiment(examples_per_class: int, seed: int, filtered_strategy: str,
                   real_dir: str, filtered_dir: str, iterations_per_epoch: int = 200, 
                   num_epochs: int = 50, batch_size: int = 32, 
                   synthetic_probability: float = 0.5, image_size: int = 256):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = FilteredPASCALDataset(
        real_dir=real_dir,
        filtered_dir=filtered_dir,
        filtered_strategy=filtered_strategy,
        split="train",
        examples_per_class=examples_per_class,
        seed=seed,
        synthetic_probability=synthetic_probability,
        image_size=(image_size, image_size)
    )

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    val_dataset = FilteredPASCALDataset(
        real_dir=real_dir,
        filtered_dir=filtered_dir,
        filtered_strategy=filtered_strategy,
        split="val",
        seed=seed,
        synthetic_probability=0,  # No synthetic images in validation
        image_size=(image_size, image_size)
    )

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Samples per epoch: {batch_size * iterations_per_epoch}")

    model = ClassificationModel(train_dataset.num_classes).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    records = []

    for epoch in trange(num_epochs, desc="Training Classifier"):
        model.train()

        epoch_loss = torch.zeros(train_dataset.num_classes, dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(train_dataset.num_classes, dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(train_dataset.num_classes, dtype=torch.float32, device='cuda')

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            if len(label.shape) > 1: label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_loss = epoch_loss / epoch_size.clamp(min=1)
        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        training_loss = training_loss.cpu().numpy()
        training_accuracy = training_accuracy.cpu().numpy()

        model.eval()

        epoch_loss = torch.zeros(train_dataset.num_classes, dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(train_dataset.num_classes, dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(train_dataset.num_classes, dtype=torch.float32, device='cuda')

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy()
        validation_accuracy = validation_accuracy.cpu().numpy()

        records.extend([
            {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": training_loss.mean(), "metric": "Loss", "split": "Training"},
            {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": validation_loss.mean(), "metric": "Loss", "split": "Validation"},
            {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": training_accuracy.mean(), "metric": "Accuracy", "split": "Training"},
            {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": validation_accuracy.mean(), "metric": "Accuracy", "split": "Validation"}
        ])

        for i, name in enumerate(train_dataset.class_names):
            records.extend([
                {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": training_loss[i], "metric": f"Loss {name.title()}", "split": "Training"},
                {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": validation_loss[i], "metric": f"Loss {name.title()}", "split": "Validation"},
                {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": training_accuracy[i], "metric": f"Accuracy {name.title()}", "split": "Training"},
                {"seed": seed, "examples_per_class": examples_per_class, "epoch": epoch, "value": validation_accuracy[i], "metric": f"Accuracy {name.title()}", "split": "Validation"}
            ])

    return records


def get_filtered_strategies(filtered_dir: str):
    return [dir for dir in os.listdir(filtered_dir) if os.path.isdir(os.path.join(filtered_dir, dir))]

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for seed in SEEDS:
        for epc in EXAMPLES_PER_CLASS:
            real_dir = os.path.join(BASE_DIR, f"pascal-real-{seed}-{epc}")
            filtered_dir = os.path.join(BASE_DIR, f"filtered-pascal-{seed}-{epc}")
            
            if not os.path.exists(real_dir):
                print(f"Warning: {real_dir} does not exist. Skipping examples_per_class={epc} for seed={seed}")
                continue

            if not os.path.exists(filtered_dir):
                print(f"Warning: {filtered_dir} does not exist. Skipping examples_per_class={epc} for seed={seed}")
                continue

            filtered_strategies = get_filtered_strategies(filtered_dir)
            
            if not filtered_strategies:
                print(f"Warning: No filtered strategies found for examples_per_class={epc} and seed={seed}. Skipping.")
                continue

            for filtered_strategy in filtered_strategies:
                print(f"Running experiment: seed={seed}, examples_per_class={epc}, filtered_strategy={filtered_strategy}")
                
                records = run_experiment(
                    examples_per_class=epc,
                    seed=seed,
                    filtered_strategy=filtered_strategy,
                    real_dir=real_dir,
                    filtered_dir=filtered_dir,
                    iterations_per_epoch=200,
                    num_epochs=50,
                    batch_size=32,
                    synthetic_probability=0.5,
                    image_size=256
                )

                df = pd.DataFrame(records)
                output_file = f"{filtered_strategy.replace('.', '_')}.csv"
                df.to_csv(os.path.join(RESULTS_DIR, output_file), index=False)
                print(f"Results saved to {output_file}")
# Smart Image Augmentation with Diffusion models and LLMs

A comprehensive toolkit for enhancing image classification performance through intelligent data augmentation, especially valuable for scenarios with limited training data.

## 📋 Overview

This project implements a state-of-the-art approach to image augmentation using diffusion models and intelligent filtering strategies. By combining traditional data augmentation techniques with AI-driven methods, it dramatically improves classification accuracy even when only a handful of examples per class are available.

## ✨ Core Features

- **Smart Augmentation**: Generate high-quality synthetic training images using diffusion models
- **Intelligent Filtration**: Automatically evaluate and filter synthetic images based on quality, diversity, and relevance
- **Customizable Strategies**: Apply various filtration strategies optimized for different data scenarios
- **Experiment Framework**: Easily run, track, and compare classification performance across different methods
- **Few-shot Learning**: Achieve excellent results with as few as 1-8 examples per class

## 🔍 How It Works

1. **Data Preparation**: Transform and organize your image dataset
2. **Image Generation**: Create diverse synthetic images using state-of-the-art diffusion models
3. **Quality Assessment**: Evaluate generated images across multiple quality dimensions
4. **Strategic Filtering**: Apply optimized filtering strategies to keep only the most valuable synthetic images
5. **Model Training**: Train classification models with the enhanced dataset
6. **Performance Evaluation**: Compare results against baselines

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/username/smart-image-augmentation.git
cd smart-image-augmentation

# Create and activate conda environment
conda create -n da-fusion python=3.8 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch
conda activate da-fusion

# Install dependencies
pip install -r requirements.txt
pip install -e ./smart-image-augmentation/generator
```

## 🚀 Getting Started

### 1. Prepare Your Dataset

```bash
python prepare_pascal.py
```

### 2. Run Baseline Experiments

```bash
# Baseline with classic augmentation
python generator/train_classifier.py \
--logdir ./pascal-baselines/baseline \
--dataset pascal --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16
```

### 3. Generate and Filter Augmented Images

```bash
# Generate augmented images using DA-Fusion
python generator/train_classifier.py \
--logdir ./pascal-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--synthetic-dir "./aug/textual-inversion-1.0-0.75-0.5-0.25/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" \
--aug textual-inversion textual-inversion textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 7.5 7.5 --strength 1.0 0.75 0.5 0.25 \
--mask 0 0 0 0 --inverted 0 0 0 0 \
--probs 0.25 0.25 0.25 0.25 --compose parallel \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16
```

### 4. Run Experiments with Filtered Images

```bash
# Run classification with filtered images
python run_experiment.py
```

## 📊 Results Visualization

```bash
# Compare methods and visualize results
python plot.py --logdirs ./pascal-baselines/baseline_randaugment ./pascal-baselines/real-guidance-0.5-cap ./pascal-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--datasets Pascal --method-dirs baseline real-guidance ours \
--method-names "RandAugment (Cubuk et al. 2020)" "Real Guidance (He et al. 2022)" "DA-Fusion (Ours)" \
--name pascal_results
```

## 🧠 Advanced Usage: Custom Filtration Strategies

The project includes multiple image filtration strategies that can be customized:

- **Top-N Selection**: Keep the best N% of images
- **Percentile Filtering**: Filter based on quality score percentiles
- **Z-Score Filtering**: Remove outliers based on statistical measures
- **Combined Strategies**: Apply multiple filters sequentially

Example configuration:

```python
strategies = [
    {'type': 'zscore_top_n', 'zscore_threshold': 2, 'top_n_value': 0.8, 
     'group_by_class': True, 'name': 'zscore_top_n_class', 'creative': True},
    {'type': 'percentile', 'value': 0.1, 'group_by_class': False, 
     'name': 'percentile_overall', 'creative': False}
]
```

## 🔧 System Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- OpenAI API key (for quality assessment)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
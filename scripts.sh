#!/bin/bash

conda create -n da-fusion python=3.8 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch

conda activate da-fusion

pip install -r requirements.txt

pip install -e ./smart-image-augmentation/generator



## Baseline with classic data augmentations technique
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/pascal-baselines/baseline
--dataset pascal --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16

## Baseline with RandAugment technique
python generator/train_classifier.py \
--logdir ./randaugment-pascal-baselines/randagument \
--dataset pascal --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16 --use-randaugment

## Real Guidance script from base paper
python generator/train_classifier.py --logdir pascal-baselines/real-guidance-0.5-cap \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/\
real-guidance-0.5-cap/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo" \
--aug real-guidance --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16

## Base plotting script
python plot.py --logdirs ./pascal-baselines/baseline_randaugment ./pascal-baselines/real-guidance-0.5-cap ./pascal-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--datasets Pascal --method-dirs baseline real-guidance ours --method-names "RandAugment (Cubuk et al. 2020)" "Real Guidance (He et al. 2022)" "DA-Fusion (Ours)" \
--name pascal_results

## Main da-fusion script from paper
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/pascal-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/textual-inversion-1.0-0.75-0.5-0.25/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" --aug textual-inversion textual-inversion textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 7.5 7.5 --strength 1.0 0.75 0.5 0.25 --mask 0 0 0 0 --inverted 0 0 0 0 \
--probs 0.25 0.25 0.25 0.25 --compose parallel --num-synthetic 10 --synthetic-probability 0.5 --num-trials 8 --examples-per-class 1 2 4 8 16

## Modified for smart-image-augmentation script
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/pascal-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/textual-inversion-1.0-0.75-0.5-0.25/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" --aug textual-inversion textual-inversion textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 7.5 7.5 --strength 1.0 0.75 0.5 0.25 --mask 0 0 0 0 --inverted 0 0 0 0 \
--probs 0.25 0.25 0.25 0.25 --compose parallel --num-synthetic 10 --synthetic-probability 0.5 --num-trials 8 --examples-per-class 1 2 4 8 16 \
--mappings-output-dir /home/ubuntu/EzLogz/smart-image-augmentation/map-images-pascal



## COCO Commands


## Baseline with classic data augmentations technique
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/coco-baselines/baseline
--dataset coco --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16

## Baseline with RandAugment technique
python generator/train_classifier.py \
--logdir ./coco-baselines/randagument \
--dataset coco --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16 --use-randaugment

## Real Guidance script from base paper
python generator/train_classifier.py --logdir coco-baselines/real-guidance-0.5-cap \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/\
real-guidance-0.5-cap/{dataset}-{seed}-{examples_per_class}" \
--dataset coco --prompt "a photo" \
--aug real-guidance --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16

## Modified for smart-image-augmentation script
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/coco-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/textual-inversion-1.0-0.75-0.5-0.25/{dataset}-{seed}-{examples_per_class}" \
--dataset coco --prompt "a photo of a {name}" --aug textual-inversion textual-inversion textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 7.5 7.5 --strength 1.0 0.75 0.5 0.25 --mask 0 0 0 0 --inverted 0 0 0 0 \
--probs 0.25 0.25 0.25 0.25 --compose parallel --num-synthetic 10 --synthetic-probability 0.5 --num-trials 8 --examples-per-class 1 2 4 8 16 \
--mappings-output-dir /home/ubuntu/EzLogz/smart-image-augmentation/map-images-coco
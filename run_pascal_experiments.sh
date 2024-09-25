#!/bin/bash

# Baseline with classic data augmentations technique
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/pascal-baselines/baseline \
--dataset pascal --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16

# Baseline with RandAugment technique
python generator/train_classifier.py \
--logdir ./randaugment-pascal-baselines/randagument \
--dataset pascal --num-synthetic 0 \
--synthetic-probability 0.0 --num-trials 8 \
--examples-per-class 1 2 4 8 16 --use-randaugment

# Real Guidance script from base paper
python generator/train_classifier.py --logdir pascal-baselines/real-guidance-0.5-cap \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/real-guidance-0.5-cap/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo" \
--aug real-guidance --guidance-scale 7.5 \
--strength 0.5 --mask 0 --inverted 0 \
--num-synthetic 10 --synthetic-probability 0.5 \
--num-trials 8 --examples-per-class 1 2 4 8 16

# Modified for smart-image-augmentation script
python generator/train_classifier.py \
--logdir /home/ubuntu/EzLogz/smart-image-augmentation/pascal-baselines/textual-inversion-1.0-0.75-0.5-0.25 \
--synthetic-dir "/home/ubuntu/EzLogz/smart-image-augmentation/aug/textual-inversion-1.0-0.75-0.5-0.25/{dataset}-{seed}-{examples_per_class}" \
--dataset pascal --prompt "a photo of a {name}" --aug textual-inversion textual-inversion textual-inversion textual-inversion \
--guidance-scale 7.5 7.5 7.5 7.5 --strength 1.0 0.75 0.5 0.25 --mask 0 0 0 0 --inverted 0 0 0 0 \
--probs 0.25 0.25 0.25 0.25 --compose parallel --num-synthetic 10 --synthetic-probability 0.5 --num-trials 8 --examples-per-class 1 2 4 8 16 \
--mappings-output-dir /home/ubuntu/EzLogz/smart-image-augmentation/map-images-pascal
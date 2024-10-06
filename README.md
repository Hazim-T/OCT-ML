# OCT Scan Classification - Dry vs Wet Macular Degeneration

This project classifies Optical Coherence Tomography (OCT) scans into **Dry** or **Wet** Macular Degeneration using a 3D Convolutional Neural Network.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)


## Project Overview
The goal of this project is to classify OCT scans to assist in the diagnosis of **Macular Degeneration**, an eye disease that leads to vision loss. OCT volumes are composed of multiple slices of the eye’s fundus, and the project uses deep learning to classify whether the patient has **Dry** or **Wet** Macular Degeneration.

## Dataset
The input data consists of **OCT volumes** stored in `.e2e` files. Each OCT scan contains multiple 2D slices spaced apart.
The dataset needs to be preprocessed using the `eyepy` library to read the `.e2e` files and create 3D tensors for training.

---
title: "MRI Brain Tumor Segmentation"
slug: "mri-tumor-segmentation"
date: "2025-06-14"
coverImage: "/images/projects/mri-tumor-segmentation.jpg"
tags: ["Deep Learning", "Medical Imaging", "Computer Vision"]
excerpt: "Three U-Net variants compared for brain tumor segmentation on 2D slices of BraTS-Africa MRI (four modalities), with a Streamlit app to compare predictions with the ground truth."
category: "Research"
githubLink: "https://github.com/Ahmad-Nayfeh/MRI-Tumor-Segmentation"
liveDemoUrl: "https://mri-tumor-segmentation-nahmad.streamlit.app/"
featured: true
status: "Completed"
highlights:
  - "U-Net with a pretrained ResNet34 encoder: 0.963 Dice"
  - "Same U-Net from scratch: 0.506 Dice"
  - "Slice-level split: an optimistic estimate"
result: "Pretrained-encoder U-Net: 0.963 test Dice (slice-level split)"
---

# MRI Brain Tumor Segmentation

> How much does a pretrained encoder matter when segmenting brain tumors from a small MRI dataset?

[Open the app](https://mri-tumor-segmentation-nahmad.streamlit.app/)

![The four MRI modalities and the tumor mask, slice by slice](/images/projects/mri-tumor-segmentation/dataset.gif)

## Setup

- **Data:** BraTS-Africa, 146 patients (one was skipped for a corrupt file). Each scan has four MRI modalities (T1, T1 with contrast, T2, FLAIR) and an expert tumor mask.
- **From 3D to 2D:** the four modalities are stacked as four input channels, each normalised to [0, 1]. Only slices that contain tumor are kept, which gives **10,692** 2D slices.
- **Split:** 70% train, 15% validation, 15% test (1,604 test slices).
- **Loss:** binary cross-entropy plus Dice loss, so that small tumors are not drowned out by background pixels.
- **Models:**
  - **ResNetUNet:** a U-Net with a ResNet34 encoder pretrained on ImageNet.
  - **BaselineUNet:** a standard U-Net trained from scratch.
  - **TransUNet (simplified):** a CNN-only stand-in for TransUNet, without the transformer blocks yet.

## Results

| Model | Test Dice |
|---|---|
| ResNetUNet (pretrained encoder) | **0.963** |
| BaselineUNet (from scratch) | 0.506 |
| TransUNet (simplified) | 0.069 |

![Training loss per model](/images/projects/mri-tumor-segmentation/loss_plot.png)

## What the numbers actually say

- **The pretrained encoder is the difference.** With the same decoder and data, starting from ImageNet weights took Dice from 0.51 to 0.96.
- **The simplified TransUNet failed,** even though its training loss converged. Without the transformer blocks and with weaker skip connections, it did not learn the spatial detail the task needs. It is a starting point, not a result.
- **The test score is optimistic.** The split was made per slice, not per patient, so neighbouring slices of the same patient appear in both training and test sets, and they look almost identical. The test also contains only slices with tumor, so false alarms on healthy slices are not measured. A per-patient split, with healthy slices included, is the honest next step and would likely give a lower score.
- **This is a learning project, not a clinical tool.** Nothing here has been validated for medical use.

![Comparing a prediction with the ground truth in the app](/images/projects/mri-tumor-segmentation/demo.png)

## Built with

Python, PyTorch and torchvision (pretrained ResNet34), NiBabel for the NIfTI scans, NumPy, and Streamlit for the app. Four notebooks cover exploration, preprocessing, training and benchmarking, and inference.

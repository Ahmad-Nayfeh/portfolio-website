---
title: "AI-Powered MRI Brain Tumor Segmentation"
slug: "mri-tumor-segmentation"
date: "2025-06-14"
coverImage: "/images/projects/mri-tumor-segmentation.jpg"
tags: ["MRI", "Deep Learning", "Image Processing", "U-Net"]
excerpt: "An end-to-end deep learning solution for the automated segmentation of brain tumors from multi-modal MRI scans."
category: "MRI Scans"
githubLink: "https://github.com/ahmad-nayfeh/MRI-Tumor-Segmentation"
liveDemoUrl: "https://mri-tumor-segmentation-nahmad.streamlit.app/"
featured: true
---

# 🧠 AI-Powered MRI Brain Tumor Segmentation

> A comprehensive deep learning project that transforms medical imaging workflow - from research to production deployment.

[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-blue?style=for-the-badge)](https://mri-tumor-segmentation-nahmad.streamlit.app/)
[![GitHub](https://img.shields.io/badge/📁-Source%20Code-black?style=for-the-badge)](https://github.com/ahmad-nayfeh/MRI-Tumor-Segmentation)

![Dataset Overview](/images/projects/mri-tumor-segmentation/dataset.gif)

## Project Overview

This project tackles one of healthcare's most critical challenges: **brain tumor diagnosis**. I developed an end-to-end AI solution that can segment brain tumors from MRI scans in under 5 minutes - a process that typically takes radiologists 30 minutes to several hours.

**The Impact:** My best model achieves a 96.34% Dice score, significantly outperforming typical human expert consistency (73-85%) and demonstrating the potential to revolutionize medical imaging workflows.

## The Problem I Solved

### Healthcare Challenge
Brain tumor segmentation is crucial for diagnosis and treatment planning, but the current manual process has serious limitations:

- **Time-Intensive**: Radiologists spend 30 minutes to several hours per scan
- **Inconsistent Results**: Human experts achieve only 73-85% consistency (Dice scores)
- **Clinical Bottleneck**: Delays in diagnosis and treatment planning
- **High Stakes**: Glioblastoma has only a 6.9% five-year survival rate - accuracy and speed matter

### My Solution
I built a comprehensive AI pipeline that delivers:
- ⚡ **95% faster processing**: 5 minutes vs. hours
- 🎯 **Superior accuracy**: 96.34% consistency vs. 73-85% human
- 🔄 **End-to-end automation**: From raw MRI to deployable web app
- 🌐 **Production-ready**: Interactive Streamlit application

## Technical Approach & Architecture

### Deep Learning Models
I implemented and rigorously benchmarked three distinct architectures, each representing different approaches to medical image segmentation:

| Model | My Implementation | Results | Key Insight |
|-------|------------------|---------|-------------|
| **ResNetUNet** | Enhanced U-Net with pre-trained ResNet34 backbone | **96.34% Dice Score** | Transfer learning dominates |
| **BaselineUNet** | Standard U-Net built from scratch | 50.59% Dice Score | Good foundation, limited generalization |
| **TransUNet** | Simplified CNN-Transformer hybrid foundation | 6.86% Dice Score* | Architectural complexity challenges |

*My TransUNet implementation was deliberately simplified as a stepping stone toward full Vision Transformer integration - the low performance revealed critical insights about skip connections and architectural requirements.

### Advanced Data Engineering Pipeline
**Dataset**: BraTS-Africa (146 patients, 4 MRI modalities each)
- **Multi-modal Integration**: Stacked T1, T1c, T2, and FLAIR sequences into 4-channel volumes
- **Smart Preprocessing**: Automated cropping of empty "air" slices, reducing dataset by ~30%
- **Normalization Strategy**: Min-max scaling across modalities for training stability
- **3D-to-2D Conversion**: Generated 10,692 training slices from 146 3D volumes
- **Data Integrity**: Rigorous 70/15/15 train/validation/test split maintaining patient-level separation

## Key Technical Achievements

### 🏗️ **End-to-End Pipeline Development**
Built complete workflow from raw medical data to production deployment:
- Data preprocessing and augmentation
- Model training with multiple architectures
- Performance benchmarking and validation
- Web application development and deployment

### 🧠 **Advanced Medical AI Implementation**
- **Multi-Modal Fusion**: Expertly handled 4 MRI sequences (T1, T1c, T2, FLAIR) with strategic channel stacking
- **Loss Function Engineering**: Designed hybrid BCE + Dice Loss to handle severe class imbalance and optimize for clinical metrics
- **Transfer Learning Mastery**: Leveraged pre-trained ResNet34 backbone, achieving 46% performance improvement over from-scratch training
- **Architecture Analysis**: Conducted thorough failure analysis revealing critical insights about skip connections and model convergence

### 🚀 **Production Deployment**
Created an interactive web application that allows real-time tumor segmentation:
- User-friendly interface for medical professionals
- Real-time model inference
- Visual comparison of different architectures

![Demo Screenshot](/images/projects/mri-tumor-segmentation/demo.png)

**[Try the Live Demo →](https://mri-tumor-segmentation-nahmad.streamlit.app/)**

## Development Process & Skills Demonstrated

### Research & Analysis
- Conducted comprehensive benchmarking of 3 distinct architectures (U-Net variants + Transformer hybrid)
- Performed rigorous quantitative and qualitative analysis of model failures
- Identified key insights: transfer learning superiority, skip connection necessity, generalization challenges
- Analyzed 10,692 2D slices across 146 patients with multi-modal MRI data

### Technical Implementation
- **Deep Learning**: PyTorch, U-Net, ResNet34 transfer learning, custom loss functions
- **Medical Data Processing**: NIfTI format handling, 3D-to-2D conversion, multi-modal fusion
- **Performance Engineering**: BCE+Dice hybrid loss, class imbalance handling, training optimization
- **Model Analysis**: Comprehensive failure analysis, architectural insights, convergence studies

### Project Management
- Structured development using Jupyter notebooks for reproducibility
- Clear documentation and code organization
- Version control and collaborative development practices

## Results & Impact

![Training Results](/images/projects/mri-tumor-segmentation/loss_plot.png)

### Quantitative Results
- **96.34% Dice Score**: ResNetUNet achieves state-of-the-art performance
- **46% improvement**: Transfer learning vs. from-scratch training (96.34% vs 50.59%)
- **10,692 samples processed**: Successfully scaled from 146 3D volumes
- **Expert-level consistency**: Exceeds human radiologist agreement (73-85%) by 10-20%

### Key Technical Discoveries
**Transfer Learning Dominance**: Pre-trained ResNet34 backbone dramatically outperformed from-scratch training, demonstrating the power of leveraging established visual features.

**Architectural Insights**: My TransUNet failure analysis revealed critical requirements for medical segmentation - specifically the necessity of skip connections for preserving spatial detail during decoder reconstruction.

**Loss Function Engineering**: The hybrid BCE+Dice loss successfully balanced pixel-level accuracy with structural similarity, directly optimizing for the clinical evaluation metric.

### Technical Learning Outcomes
- Mastered medical AI domain-specific challenges
- Gained expertise in U-Net and advanced CNN architectures
- Developed skills in 3D medical data processing
- Created production-ready ML applications

### Potential Real-World Impact
This project demonstrates how AI can transform healthcare delivery:
- **Clinical Efficiency**: Dramatically reduces radiologist workload
- **Diagnostic Consistency**: Eliminates human variability in critical diagnoses
- **Accessibility**: Makes expert-level analysis available globally
- **Treatment Planning**: Provides precise tumor boundaries for radiation therapy

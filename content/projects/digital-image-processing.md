---
title: "Digital Image Processing Projects"
slug: "digital-image-processing"
date: "2024-05-01"
coverImage: "/images/projects/digital-image-processing.jpg"
tags: ["Image Processing", "Educational Tools", "Denoising", "Edge Detection", "Visualization"]
excerpt: "A modular suite of classic image processing mini-projects — covering interpolation, compression, denoising, enhancement, and shading correction — with clean visual benchmarks and Python implementations."
category: "Computer Vision"
githubLink: "https://github.com/nahmad2000/Digital-Image-Processing"
challenge: "Building a real-time collaborative task management system that works seamlessly across devices was the main challenge. The app needed to handle concurrent updates from multiple users and provide a smooth, responsive experience."
solution: "I used Firebase for real-time database and authentication, which allowed for instant updates across all connected clients. React was used for the UI, with TypeScript providing type safety and improved developer experience."
technologies:
  - "React"
  - "Firebase (Firestore, Authentication)"
  - "TypeScript"
  - "CSS Modules"
  - "React DnD (for drag-and-drop functionality)"
  - "React Query"
features:
  - "Task creation and management"
  - "Project organization"
  - "Due dates and reminders"
  - "Drag-and-drop interface"
  - "Real-time collaboration"
  - "User authentication and authorization"
  - "Mobile-responsive design"
featured: true
---

## 🧠 Project Summary

This repository is a curated set of image processing mini-projects developed in Python. Each subfolder tackles a key concept — such as interpolation accuracy, shading correction, or denoising — and offers an end-to-end experimental setup: from transformation to visualization. The goal is to distill theoretical concepts into practical, testable code for both learning and application.

---

## 📂 Subproject Gallery

### 🔻 1. Downsampling & Interpolation  
This module explores how downsampling methods (Simple, Anti-aliased, Area-based) and interpolation strategies (Nearest, Bilinear, Bicubic, Lanczos) affect image quality. It generates full grids of upsampled results and evaluates combinations using SSIM and PSNR.

Results show that Area-based downsampling followed by Lanczos interpolation provides the best balance of detail and structural fidelity. The module also auto-generates heatmaps to compare metric scores visually.

![Downsampling heatmap metrics](/images/projects/digital-image-processing/dip-downsampling.png)

---

### 🔄 2. Geometric Transformations  
Applies core affine transformations in batch — including rotation, scaling, translation, and shearing — with a flexible command-line interface. All operations are applied using transformation matrices and OpenCV’s high-performance functions.

What makes this unique is the ability to stack multiple transformations in a single run, apply padding, and save annotated outputs systematically. It's ideal for preprocessing pipelines or synthetic data generation.

![Rotated Image Example](/images/projects/digital-image-processing/dip-geometry.png)

---

### 🗜️ 3. Image Compression  
Benchmarks JPEG, PNG, and WebP formats on file size vs. visual fidelity. It calculates Compression Ratio, MSE, PSNR, and SSIM across images, then visualizes results using bar charts, scatter plots, and summary CSV exports.

Includes parallel processing for speed and customizable quality levels for each format. Perfect for evaluating trade-offs in applications like web optimization or mobile image delivery.

![Compression Metric Barplots](/images/projects/digital-image-processing/dip-compression.png)

---

### 🧹 4. Image Denoising  
Simulates noisy environments using Gaussian and Salt & Pepper noise, then applies classical denoising filters — Median, Gaussian Blur, and Non-Local Means. Outputs include before/after comparisons, metric bar charts, and histograms.

It also measures how each filter performs in restoring structure (SSIM) and fidelity (PSNR), offering insight into the strengths and weaknesses of each method under controlled degradation.

![Denoising Filter Comparison](/images/projects/digital-image-processing/dip-denoising.png)

---

### ✨ 5. Image Enhancement  
Applies two key enhancement techniques — Gamma Correction (Power Law) and Histogram Equalization — to improve image brightness and contrast. Users can run it interactively or through CLI with tunable parameters like gamma value and constant multiplier.

The system even visualizes how the pixel intensity distribution changes before and after enhancement, helping to explain the effects beyond just the output image.

![Enhanced Image Comparison](/images/projects/digital-image-processing/dip-enhancement.png)

---

### 🌗 6. Shading Correction  
Corrects uneven lighting using both spatial and frequency domain techniques. The spatial method uses Gaussian blur subtraction, while the frequency method implements homomorphic filtering (with CLAHE).

It includes full visualizations of corrected images and extracted illumination layers — providing both corrective outputs and insight into the nature of lighting artifacts in images.

![Shading Correction Output](/images/projects/digital-image-processing/dip-shading.png)

---

## ⚙️ Technologies Used

- Python 3.x  
- OpenCV • NumPy • scikit-image  
- Matplotlib • Seaborn  
- Jupyter Notebooks • Command-line Interfaces  
- Parallel Processing (Compression module)

---

## 🧠 Key Takeaways

- Each subproject translates a fundamental image processing technique into hands-on, reproducible experiments.
- Visualizations make metric-based evaluation clear and digestible.
- Modular code design allows you to plug components into larger CV pipelines or teaching demos.

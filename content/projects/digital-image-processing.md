---
title: "DIP-Lib: Interactive Digital Image Processing Toolkit"
slug: "digital-image-processing-lib"
date: "2024-05-01"
coverImage: "/images/projects/digital-image-processing-lib.jpg"
tags: ["Image Processing", "Educational Tools", "Denoising", "Edge Detection", "Visualization"]
excerpt: "A modular Streamlit-based web app for exploring core image processing techniques — complete with parameterized controls, visual comparisons, and dynamic pipelines."
category: "Computer Vision"
githubLink: "https://github.com/nahmad2000/Digital-Image-Processing-Library"
liveDemoUrl: "https://digital-image-processing-library-nahmad.streamlit.app/"
featured: true
---

<div class="project-prose-container">

## 🧠 Project Summary
DIP-Lib is a hands-on, visual toolkit for digital image processing — built as an interactive web app using Streamlit. Designed for learners, researchers, and engineers alike, it brings together 9 classical image processing modules into a single pipeline-based UI where users can adjust parameters, stack transformations, and observe results live.

<br/>
<a href="https://digital-image-processing-library-nahmad.streamlit.app/" target="_blank" rel="noopener noreferrer" class="project-cta-link">
    🚀 Launch the App
</a>

## 📂 Core Modules

### 🔻 1. Downsampling & Interpolation
Resize images with different downsampling methods (`simple`, `antialias`, `area`) and upscale them using interpolation (`nearest`, `bilinear`, `bicubic`, `lanczos`). DIP-Lib visualizes how each combination affects quality using PSNR and SSIM metrics.

### 🔄 2. Geometric Transformations
Apply affine and projective transformations such as rotation, scaling, translation, and shearing. Parameters are controlled via sliders, and transformations can be layered and previewed dynamically.

### 🧹 3. Noise Analysis & Removal
Add synthetic noise (Gaussian or Salt & Pepper) and evaluate denoising filters:
- **Median Filter** (great for impulse noise)
- **Gaussian Blur** (for smoothing)
- **Non-Local Means** (for preserving texture)

### ✨ 4. Image Enhancement
Boost brightness and contrast using **Gamma Correction**, **Histogram Equalization**, and **CLAHE** (adaptive enhancement) to optimize visibility in dark or low-contrast images.

### 🔬 6. Edge Detection & Sharpening
Compare classic edge detectors like **Sobel**, **Scharr**, **Laplacian**, and **Canny**. Also applies **Unsharp Masking** to enhance fine details.

## 🛠️ Technologies Used
- Streamlit (UI framework)
- OpenCV (image manipulation)
- NumPy, scikit-image
- Matplotlib, Seaborn

## 📸 Sample Gallery
<div class="project-gallery">
    <img src="/images/projects/digital-image-processing-lib/demo1.png" alt="Demo Screenshot 1" />
    <img src="/images/projects/digital-image-processing-lib/demo2.png" alt="Demo Screenshot 2" />
    <img src="/images/projects/digital-image-processing-lib/demo3.png" alt="Demo Screenshot 3" />
</div>

## 💡 Why It Matters
- **Educational**: Helps students learn through visual experimentation.
- **Research Utility**: Benchmark filters, compare transformations, or prep data.
- **Usability-first**: Web-based, real-time, no installation required.

</div>
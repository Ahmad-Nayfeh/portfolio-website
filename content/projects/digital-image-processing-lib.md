---
title: "DIP-Lib: Interactive Digital Image Processing Toolkit"
slug: "digital-image-processing-lib"
date: "2025-04-30"
coverImage: "/images/projects/digital-image-processing-lib.jpg"
tags: ["Image Processing", "Computer Vision"]
excerpt: "A modular Streamlit-based web app for exploring core image processing techniques — complete with parameterized controls, visual comparisons, and dynamic pipelines."
category: "Lab"
githubLink: "https://github.com/Ahmad-Nayfeh/Digital-Image-Processing-Library"
liveDemoUrl: "https://digital-image-processing-library-nahmad.streamlit.app/"
featured: false
---

<div class="project-prose-container">

## 🧠 Project Summary
**DIP-Lib** is a hands-on, visual toolkit for digital image processing, built as an interactive web app using Streamlit. It's designed for learners and researchers to explore 9 classical image processing modules in a single, pipeline-based UI. You can adjust parameters, stack transformations, and see the results live, helping build intuition through guided, real-time feedback.

<br/>
<a href="https://digital-image-processing-library-nahmad.streamlit.app/" target="_blank" rel="noopener noreferrer" class="project-cta-link">
    🚀 Launch the Interactive App
</a>

## 📂 Core Modules
The toolkit is divided into distinct, interactive modules:

### 🔻 1. Downsampling & Interpolation
Resize images with methods like `simple` or `antialias` and upscale using `nearest`, `bilinear`, or `lanczos` interpolation. The app visualizes how each combination affects quality using **PSNR** and **SSIM** metrics.

### 🔄 2. Geometric Transformations
Apply affine and projective transformations like rotation, scaling, and shearing. All parameters are controlled via sliders and can be layered dynamically.

### 🧹 3. Noise Analysis & Removal
Add synthetic **Gaussian** or **Salt & Pepper** noise, then evaluate various denoising filters:
- **Median Filter** (great for impulse noise)
- **Gaussian Blur** (for smoothing)
- **Non-Local Means** (excellent for preserving texture)

### ✨ 4. Image Enhancement
Boost brightness and contrast using Gamma Correction, Histogram Equalization, and **CLAHE** (adaptive enhancement) to optimize visibility in dark or low-contrast images.

### 🌗 5. Lighting Correction
Fix non-uniform lighting with two powerful techniques: **spatial filtering** (Gaussian blur subtraction) and **homomorphic filtering** (frequency domain suppression).

### 🔬 6. Edge Detection & Sharpening
Compare classic edge detectors like **Sobel**, **Scharr**, **Laplacian**, and **Canny**. This module also includes an **Unsharp Masking** filter to enhance fine details.

## 🛠️ Technologies Used
- **UI Framework**: Streamlit
- **Core Processing**: OpenCV, NumPy, scikit-image
- **Visualization**: Matplotlib, Seaborn

## 📸 Sample Gallery
<div class="project-gallery">
    <img src="/images/projects/digital-image-processing-lib/demo1.png" alt="DIP-Lib Demo 1" />
    <img src="/images/projects/digital-image-processing-lib/demo2.png" alt="DIP-Lib Demo 2" />
    <img src="/images/projects/digital-image-processing-lib/demo3.png" alt="DIP-Lib Demo 3" />
</div>

## 💡 Why It Matters
- **Educational**: Helps students learn through direct, visual experimentation.
- **Research Utility**: Useful for benchmarking filters, comparing transformations, and prepping data.
- **Usability-First**: Completely web-based with real-time feedback and no installation required.

</div>
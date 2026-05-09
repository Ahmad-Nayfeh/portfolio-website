---
title: "Digital Image Processing Projects"
slug: "digital-image-processing"
date: "2024-05-01"
coverImage: "/images/projects/digital-image-processing.jpg"
tags: ["Image Processing", "Computer Vision"]
excerpt: "A modular suite of classic image processing mini-projects — covering interpolation, compression, denoising, enhancement, and shading correction — with clean visual benchmarks and Python implementations."
category: "Computer Vision"
githubLink: "https://github.com/nahmad2000/Digital-Image-Processing"
featured: false
---

<div class="project-prose-container">

## 🧠 Project Summary
This repository is a curated set of image processing mini-projects developed in Python. Each subfolder tackles a key concept — such as interpolation accuracy, shading correction, or denoising — and offers an end-to-end experimental setup: from transformation to visualization. The goal is to distill theoretical concepts into practical, testable code for both learning and application.

## 📂 Subproject Gallery
<div class="project-gallery">
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-downsampling.png" alt="Downsampling heatmap metrics" />
        <div class="dip-card-content">
            <h3>🔻 1. Downsampling & Interpolation</h3>
            <p>This module explores how downsampling methods (Simple, Anti-aliased, Area-based) and interpolation strategies affect image quality, evaluating combinations using SSIM and PSNR.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-geometry.png" alt="Rotated Image Example" />
        <div class="dip-card-content">
            <h3>🔄 2. Geometric Transformations</h3>
            <p>Applies core affine transformations in batch — including rotation, scaling, translation, and shearing — with a flexible command-line interface. Ideal for preprocessing pipelines.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-compression.png" alt="Compression Metric Barplots" />
        <div class="dip-card-content">
            <h3>🗜️ 3. Image Compression</h3>
            <p>Benchmarks JPEG, PNG, and WebP formats on file size vs. visual fidelity. It calculates and visualizes Compression Ratio, MSE, PSNR, and SSIM across images.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-denoising.png" alt="Denoising Filter Comparison" />
        <div class="dip-card-content">
            <h3>🧹 4. Image Denoising</h3>
            <p>Simulates noisy environments and applies classical denoising filters (Median, Gaussian Blur, NLM), providing detailed before/after comparisons and metrics.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-enhancement.png" alt="Enhanced Image Comparison" />
        <div class="dip-card-content">
            <h3>✨ 5. Image Enhancement</h3>
            <p>Applies Gamma Correction and Histogram Equalization to improve image brightness and contrast, with visualizations of the pixel intensity distribution.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-shading.png" alt="Shading Correction Output" />
        <div class="dip-card-content">
            <h3>🌗 6. Shading Correction</h3>
            <p>Corrects uneven lighting using both spatial (Gaussian blur subtraction) and frequency domain (homomorphic filtering) techniques.</p>
        </div>
    </div>
</div>

## ⚙️ Technologies Used
- Python 3.x
- OpenCV • NumPy • scikit-image
- Matplotlib • Seaborn
- Jupyter Notebooks • Command-line Interfaces
- Parallel Processing (Compression module)

## 🧠 Key Takeaways
- Each subproject translates a fundamental image processing technique into hands-on, reproducible experiments.
- Visualizations make metric-based evaluation clear and digestible.
- Modular code design allows you to plug components into larger CV pipelines or teaching demos.

</div>
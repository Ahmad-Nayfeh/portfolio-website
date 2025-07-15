---
title: "Digital Image Processing Projects"
slug: "digital-image-processing"
date: "2024-05-01"
coverImage: "/images/projects/digital-image-processing.jpg"
tags: ["Image Processing", "Educational Tools", "Denoising", "Edge Detection", "Visualization"]
excerpt: "A modular suite of classic image processing mini-projects — covering interpolation, compression, denoising, enhancement, and shading correction — with clean visual benchmarks and Python implementations."
category: "Computer Vision"
githubLink: "https://github.com/nahmad2000/Digital-Image-Processing"
featured: false
---

<style>
  /* Main container for this specific project page */
  .dip-project-container {
    font-family: 'Inter', sans-serif;
    line-height: 1.75;
  }

  /* Styling for the main H1 title (already handled by the template, but we can override) */
  .dip-project-container h1 {
    text-align: center;
    font-size: 2.75rem; /* 44px */
    font-weight: 800;
    letter-spacing: -0.02em;
    border-bottom: 2px solid hsl(var(--primary) / 0.1);
    padding-bottom: 1rem;
    margin-bottom: 2rem;
  }

  /* Styling for all H2 subheadings */
  .dip-project-container h2 {
    font-size: 1.75rem; /* 28px */
    font-weight: 700;
    color: hsl(var(--primary));
    margin-top: 3rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid hsl(var(--border));
  }
  
  /* Styling for all H3 subheadings inside the gallery */
  .dip-project-container h3 {
      font-size: 1.25rem;
      font-weight: 600;
      margin-top: 0;
      margin-bottom: 0.75rem;
  }
  
  /* Styling for paragraphs */
  .dip-project-container p {
      font-size: 1.1rem;
      color: hsl(var(--muted-foreground));
  }

  /* Styling for lists */
  .dip-project-container ul {
    list-style-type: none;
    padding-left: 0;
  }
  .dip-project-container li {
    background-color: hsl(var(--secondary));
    padding: 0.75rem 1.25rem;
    border-radius: 0.5rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid hsl(var(--primary));
    transition: all 0.2s ease-in-out;
  }
  .dip-project-container li:hover {
      transform: translateX(4px);
      border-left-width: 6px;
  }

  /* Gallery specific styles */
  .dip-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
  }
  .dip-card {
    background: hsl(var(--card));
    border: 1px solid hsl(var(--border));
    border-radius: 0.75rem;
    overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  .dip-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  }
  .dip-card img {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-bottom: 1px solid hsl(var(--border));
  }
  .dip-card-content {
    padding: 1.5rem;
  }
  .dip-card-content p {
    font-size: 1rem;
    line-height: 1.6;
  }
</style>

<div class="dip-project-container">

## 🧠 Project Summary

This repository is a curated set of image processing mini-projects developed in Python. Each subfolder tackles a key concept — such as interpolation accuracy, shading correction, or denoising — and offers an end-to-end experimental setup: from transformation to visualization. The goal is to distill theoretical concepts into practical, testable code for both learning and application.

## 📂 Subproject Gallery

<div class="dip-gallery">
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-downsampling.png" alt="Downsampling heatmap metrics" />
        <div class="dip-card-content">
            <h3>🔻 1. Downsampling & Interpolation</h3>
            <p>This module explores how downsampling methods (Simple, Anti-aliased, Area-based) and interpolation strategies (Nearest, Bilinear, Bicubic, Lanczos) affect image quality. It generates full grids of upsampled results and evaluates combinations using SSIM and PSNR. Results show that Area-based downsampling followed by Lanczos interpolation provides the best balance of detail and structural fidelity. The module also auto-generates heatmaps to compare metric scores visually.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-geometry.png" alt="Rotated Image Example" />
        <div class="dip-card-content">
            <h3>🔄 2. Geometric Transformations</h3>
            <p>Applies core affine transformations in batch — including rotation, scaling, translation, and shearing — with a flexible command-line interface. All operations are applied using transformation matrices and OpenCV’s high-performance functions. What makes this unique is the ability to stack multiple transformations in a single run, apply padding, and save annotated outputs systematically. It's ideal for preprocessing pipelines or synthetic data generation.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-compression.png" alt="Compression Metric Barplots" />
        <div class="dip-card-content">
            <h3>🗜️ 3. Image Compression</h3>
            <p>Benchmarks JPEG, PNG, and WebP formats on file size vs. visual fidelity. It calculates Compression Ratio, MSE, PSNR, and SSIM across images, then visualizes results using bar charts, scatter plots, and summary CSV exports. Includes parallel processing for speed and customizable quality levels for each format. Perfect for evaluating trade-offs in applications like web optimization or mobile image delivery.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-denoising.png" alt="Denoising Filter Comparison" />
        <div class="dip-card-content">
            <h3>🧹 4. Image Denoising</h3>
            <p>Simulates noisy environments using Gaussian and Salt & Pepper noise, then applies classical denoising filters — Median, Gaussian Blur, and Non-Local Means. Outputs include before/after comparisons, metric bar charts, and histograms. It also measures how each filter performs in restoring structure (SSIM) and fidelity (PSNR), offering insight into the strengths and weaknesses of each method under controlled degradation.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-enhancement.png" alt="Enhanced Image Comparison" />
        <div class="dip-card-content">
            <h3>✨ 5. Image Enhancement</h3>
            <p>Applies two key enhancement techniques — Gamma Correction (Power Law) and Histogram Equalization — to improve image brightness and contrast. Users can run it interactively or through CLI with tunable parameters like gamma value and constant multiplier. The system even visualizes how the pixel intensity distribution changes before and after enhancement, helping to explain the effects beyond just the output image.</p>
        </div>
    </div>
    <div class="dip-card">
        <img src="/images/projects/digital-image-processing/dip-shading.png" alt="Shading Correction Output" />
        <div class="dip-card-content">
            <h3>🌗 6. Shading Correction</h3>
            <p>Corrects uneven lighting using both spatial and frequency domain techniques. The spatial method uses Gaussian blur subtraction, while the frequency method implements homomorphic filtering (with CLAHE). It includes full visualizations of corrected images and extracted illumination layers — providing both corrective outputs and insight into the nature of lighting artifacts in images.</p>
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
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
    .project-container-grid {
        font-family: 'Inter', sans-serif;
        line-height: 1.75;
    }
    .project-header-grid {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid hsl(var(--border));
        margin-bottom: 2.5rem;
    }
    .project-header-grid h1 {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    .project-header-grid p.subtitle {
        font-size: 1.125rem;
        color: hsl(var(--muted-foreground));
        max-width: 700px;
        margin: 0 auto;
    }
    .project-section-grid {
        margin-bottom: 3rem;
    }
    .project-section-grid h2 {
        font-size: 1.75rem;
        font-weight: 700;
        color: hsl(var(--primary));
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid hsl(var(--primary) / 0.1);
    }
    .subproject-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
    }
    .subproject-card {
        background: hsl(var(--card));
        border: 1px solid hsl(var(--border));
        border-radius: 0.75rem;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .subproject-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .subproject-card img {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-bottom: 1px solid hsl(var(--border));
    }
    .subproject-content {
        padding: 1.5rem;
    }
    .subproject-content h3 {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    .subproject-content p {
        font-size: 0.95rem;
        color: hsl(var(--muted-foreground));
        margin-bottom: 0;
    }
    .styled-list ul {
        list-style-type: none;
        padding: 0;
    }
    .styled-list li {
        background-color: hsl(var(--secondary));
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid hsl(var(--primary));
    }
</style>
<div class="project-container-grid">
    <div class="project-header-grid">
        <h1>Digital Image Processing Projects</h1>
        <p class="subtitle">A modular suite of classic image processing mini-projects — covering interpolation, compression, denoising, enhancement, and shading correction — with clean visual benchmarks and Python implementations.</p>
    </div>
    <div class="project-section-grid">
        <h2>🧠 Project Summary</h2>
        <p>This repository is a curated set of image processing mini-projects developed in Python. Each subfolder tackles a key concept — such as interpolation accuracy, shading correction, or denoising — and offers an end-to-end experimental setup: from transformation to visualization. The goal is to distill theoretical concepts into practical, testable code for both learning and application.</p>
    </div>
    <div class="project-section-grid">
        <h2>📂 Subproject Gallery</h2>
        <div class="subproject-gallery">
            <div class="subproject-card">
                <img src="/images/projects/digital-image-processing/dip-downsampling.png" alt="Downsampling heatmap metrics" />
                <div class="subproject-content">
                    <h3>🔻 1. Downsampling & Interpolation</h3>
                    <p>This module explores how downsampling methods (Simple, Anti-aliased, Area-based) and interpolation strategies (Nearest, Bilinear, Bicubic, Lanczos) affect image quality. It generates full grids of upsampled results and evaluates combinations using SSIM and PSNR. Results show that Area-based downsampling followed by Lanczos interpolation provides the best balance of detail and structural fidelity. The module also auto-generates heatmaps to compare metric scores visually.</p>
                </div>
            </div>
            <div class="subproject-card">
                <img src="/images/projects/digital-image-processing/dip-geometry.png" alt="Rotated Image Example" />
                <div class="subproject-content">
                    <h3>🔄 2. Geometric Transformations</h3>
                    <p>Applies core affine transformations in batch — including rotation, scaling, translation, and shearing — with a flexible command-line interface. All operations are applied using transformation matrices and OpenCV’s high-performance functions. What makes this unique is the ability to stack multiple transformations in a single run, apply padding, and save annotated outputs systematically. It's ideal for preprocessing pipelines or synthetic data generation.</p>
                </div>
            </div>
            <div class="subproject-card">
                <img src="/images/projects/digital-image-processing/dip-compression.png" alt="Compression Metric Barplots" />
                <div class="subproject-content">
                    <h3>🗜️ 3. Image Compression</h3>
                    <p>Benchmarks JPEG, PNG, and WebP formats on file size vs. visual fidelity. It calculates Compression Ratio, MSE, PSNR, and SSIM across images, then visualizes results using bar charts, scatter plots, and summary CSV exports. Includes parallel processing for speed and customizable quality levels for each format. Perfect for evaluating trade-offs in applications like web optimization or mobile image delivery.</p>
                </div>
            </div>
            <div class="subproject-card">
                <img src="/images/projects/digital-image-processing/dip-denoising.png" alt="Denoising Filter Comparison" />
                <div class="subproject-content">
                    <h3>🧹 4. Image Denoising</h3>
                    <p>Simulates noisy environments using Gaussian and Salt & Pepper noise, then applies classical denoising filters — Median, Gaussian Blur, and Non-Local Means. Outputs include before/after comparisons, metric bar charts, and histograms. It also measures how each filter performs in restoring structure (SSIM) and fidelity (PSNR), offering insight into the strengths and weaknesses of each method under controlled degradation.</p>
                </div>
            </div>
            <div class="subproject-card">
                <img src="/images/projects/digital-image-processing/dip-enhancement.png" alt="Enhanced Image Comparison" />
                <div class="subproject-content">
                    <h3>✨ 5. Image Enhancement</h3>
                    <p>Applies two key enhancement techniques — Gamma Correction (Power Law) and Histogram Equalization — to improve image brightness and contrast. Users can run it interactively or through CLI with tunable parameters like gamma value and constant multiplier. The system even visualizes how the pixel intensity distribution changes before and after enhancement, helping to explain the effects beyond just the output image.</p>
                </div>
            </div>
            <div class="subproject-card">
                <img src="/images/projects/digital-image-processing/dip-shading.png" alt="Shading Correction Output" />
                <div class="subproject-content">
                    <h3>🌗 6. Shading Correction</h3>
                    <p>Corrects uneven lighting using both spatial and frequency domain techniques. The spatial method uses Gaussian blur subtraction, while the frequency method implements homomorphic filtering (with CLAHE). It includes full visualizations of corrected images and extracted illumination layers — providing both corrective outputs and insight into the nature of lighting artifacts in images.</p>
                </div>
            </div>
        </div>
    </div>
    <div class="project-section-grid styled-list">
        <h2>⚙️ Technologies Used</h2>
        <ul>
            <li>Python 3.x</li>
            <li>OpenCV • NumPy • scikit-image</li>
            <li>Matplotlib • Seaborn</li>
            <li>Jupyter Notebooks • Command-line Interfaces</li>
            <li>Parallel Processing (Compression module)</li>
        </ul>
    </div>
    <div class="project-section-grid styled-list">
        <h2>🧠 Key Takeaways</h2>
        <ul>
            <li>Each subproject translates a fundamental image processing technique into hands-on, reproducible experiments.</li>
            <li>Visualizations make metric-based evaluation clear and digestible.</li>
            <li>Modular code design allows you to plug components into larger CV pipelines or teaching demos.</li>
        </ul>
    </div>
</div>
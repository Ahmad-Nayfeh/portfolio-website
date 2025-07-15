---
title: "Edge Vision"
slug: "edge-vision"
date: "2024-04-01"
coverImage: "/images/projects/edge-vision.jpg"
tags: ["Edge Detection", "Image Processing", "Computer Vision", "Educational Tools", "Visualization"]
excerpt: "Compare multiple edge detection algorithms (Sobel, Scharr, Laplacian, Canny) with real-time visualization and interactive parameter control."
category: "Computer Vision"
githubLink: "https://github.com/nahmad2000/Edge-Vision"
liveDemoUrl: "https://edge-vision-nahmad.streamlit.app/"
featured: false
---

<div class="project-prose-container">

## 🔬 Edge Vision: An Interactive Comparison Tool

<br/>
<a href="https://edge-vision-nahmad.streamlit.app/" target="_blank" rel="noopener noreferrer" class="project-cta-link">
    🚀 Launch the Interactive App
</a>

<div class="project-two-col-layout">

  <div class="project-main-content">
    <h3>Project Overview</h3>
    <p>
      Edge Vision is a Streamlit-based application for visually comparing classical edge detection algorithms including Sobel, Scharr, Laplacian, and Canny. With an intuitive interface and adjustable parameters, users can analyze how different methods influence edge detection results in real-time. It serves both as a learning tool and a benchmarking sandbox for vision researchers.
    </p>

    <h3>Motivation</h3>
    <p>
      In traditional computer vision, edge detection is a foundational step for object recognition and feature extraction. However, comparing different edge detectors usually requires coding from scratch. Edge Vision solves this by offering an interactive tool where users can visually explore the strengths and limitations of each algorithm side-by-side.
    </p>
    
    <h3>Reflection</h3>
    <p>
      This project sharpened my ability to bridge algorithmic vision concepts with user-centric interfaces. It also gave me practical experience with Streamlit app design, OpenCV filters, and scalable UI/UX structuring for educational tools. In future versions, I aim to add noise robustness testing and integration of deep edge detectors for comparison.
    </p>
  </div>

  <div class="project-sidebar-content">
    <div class="project-feature-box">
        <h3>Technical Approach</h3>
        <ul>
            <li><strong>Frontend:</strong> Streamlit</li>
            <li><strong>Algorithms:</strong> OpenCV</li>
            <li><strong>Controls:</strong> Full parameter control</li>
            <li><strong>Display:</strong> Matplotlib grid layout</li>
        </ul>
    </div>
    <div class="project-feature-box" style="margin-top: 1.5rem;">
        <h3>Key Features</h3>
        <ul>
            <li>Interactive UI</li>
            <li>Side-by-side visualization</li>
            <li>Custom `compare_all()` pipeline</li>
            <li>Educational performance notes</li>
            <li>Extensible backend</li>
        </ul>
    </div>
  </div>

</div>

## 📊 Results & Gallery
<p>The application demonstrates that **Canny** consistently produces cleaner results, **Scharr** offers sharper edges than Sobel, and **Laplacian** is useful for symmetric edge detection but is sensitive to noise.</p>

<div class="project-gallery">
    <img src="/images/projects/edge-vision/demo1.png" alt="Edge Vision UI showing multiple algorithms" />
    <img src="/images/projects/edge-vision/demo2.png" alt="Edge Vision UI with parameter controls" />
</div>

</div>
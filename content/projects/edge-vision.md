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

# Edge Vision

## Overview

Edge Vision is a Streamlit-based application for visually comparing classical edge detection algorithms including Sobel, Scharr, Laplacian, and Canny. With an intuitive interface and adjustable parameters, users can analyze how different methods and settings influence edge detection results — all in real-time. It serves both as a learning tool and a benchmarking sandbox for vision researchers.

## Motivation

In traditional computer vision, edge detection is a foundational step for object recognition, segmentation, and feature extraction. However, comparing different edge detectors usually requires coding from scratch. Edge Vision solves this by offering an interactive tool where users can visually explore strengths and limitations of each algorithm side-by-side.

## Technical Approach

- **Frontend**: Streamlit web interface for easy use and deployment
- **Backbone Algorithms**: Implemented in OpenCV and wrapped in modular Python functions
- **User Controls**:
  - Upload or choose sample image
  - Toggle each algorithm on/off
  - Customize parameters: kernel size, thresholds, direction (dx/dy), and more
- **Display Engine**:
  - Edge results rendered in a grid layout using Matplotlib
  - Tabs for performance notes explaining when to use each method

## Key Features / Contributions

- Interactive UI with full parameter control for each detector
- Visualization of all outputs (Original, Sobel, Scharr, Laplacian, Canny)
- Custom `compare_all()` pipeline with clean fallback and error handling
- Educational performance notes explaining detector pros/cons
- Fully modular and extensible backend for adding new methods (e.g., Prewitt, Roberts)

## Results & Findings

- **Canny** consistently produces cleaner and more accurate results when properly tuned
- **Scharr** offers sharper edges and better gradient estimates than Sobel
- **Laplacian** is more noise-sensitive but useful for symmetric edge detection
- Performance visualization is highly responsive, even for large images


![Edge Vision UI-1](/images/projects/edge-vision/demo1.png)
![Edge Vision UI-2](/images/projects/edge-vision/demo2.png)


## Reflection

This project sharpened my ability to bridge algorithmic vision concepts with user-centric interfaces. It also gave me practical experience with Streamlit app design, OpenCV filters, and scalable UI/UX structuring for educational tools. In future versions, I aim to add noise robustness testing and integration of deep edge detectors for comparison.
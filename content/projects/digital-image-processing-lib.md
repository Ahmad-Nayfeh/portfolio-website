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

**DIP-Lib** is a hands-on, visual toolkit for digital image processing — built as an interactive web app using Streamlit. Designed for learners, researchers, and engineers alike, it brings together 9 classical image processing modules into a single pipeline-based UI where users can adjust parameters, stack transformations, and observe results live.

Whether you're experimenting with thresholding or cleaning noisy images, DIP-Lib helps you build intuition through guided exploration and real-time visual feedback.

---

## 🚀 Live Demo

👉 [Launch the DIP-Lib Web App](https://digital-image-processing-library-nahmad.streamlit.app/)  
🖥️ *(No installation needed — works directly in your browser)*

---

## 📂 Core Modules

### 🔻 1. Downsampling & Interpolation  
Resize images with different downsampling methods (`simple`, `antialias`, `area`) and upscale them using interpolation (`nearest`, `bilinear`, `bicubic`, `lanczos`). DIP-Lib visualizes how each combination affects quality using PSNR and SSIM metrics.

---

### 🔄 2. Geometric Transformations  
Apply affine and projective transformations such as rotation, scaling, translation, and shearing. Parameters are controlled via sliders, and transformations can be layered and previewed dynamically.

---

### 🧹 3. Noise Analysis & Removal  
Add synthetic noise (Gaussian or Salt & Pepper) and evaluate denoising methods:  
- **Median Filter** (great for impulse noise)  
- **Gaussian Blur** (for smoothing)  
- **Non-Local Means** (for preserving texture)  

Metrics and visual plots allow you to objectively compare filters.

---

### ✨ 4. Image Enhancement  
Boost brightness and contrast using:
- **Gamma Correction**  
- **Histogram Equalization**  
- **CLAHE** (adaptive enhancement)

Choose single or combined techniques to optimize visibility in dark or low-contrast images.

---

### 🌗 5. Lighting Correction  
Fix non-uniform lighting using:
- **Spatial filtering** (Gaussian blur background subtraction)  
- **Homomorphic filtering** (frequency domain illumination suppression)

Includes preview of both corrected and illumination-extracted images.

---

### 🧠 6. Edge Detection  
Compare classic edge detectors:
- Sobel, Scharr, Laplacian, and Canny  
- Control thresholds, aperture size, and L2 gradient

Clean edges can be better observed by first denoising the image.

---

### 🔬 7. Sharpening  
Applies **Unsharp Masking** to enhance fine details and edges using weighted Gaussian subtraction. Parameter sliders control blur kernel and sharpening intensity.

---

### ⚪ 8. Thresholding  
Convert grayscale images to binary using global, adaptive, or hybrid methods. Choose from:
- Global: `binary`, `trunc`, `tozero`, etc.  
- Adaptive: `mean` or `gaussian`  
- Advanced: Otsu and Triangle methods (auto-calculated thresholds)

---

### 🌈 9. Color Space Conversion  
Convert images between BGR, RGB, GRAY, HSV, LAB, HLS, and YCrCb. You can view the full image or isolate individual color channels for detailed inspection.

---

## 🛠️ Technologies Used

- **Streamlit** (UI framework)
- **OpenCV** (image manipulation)
- **NumPy**, **scikit-image** (processing utilities)
- **Matplotlib**, **Seaborn** (metrics visualization)
- Modular structure with:
  - `image_processors.py` (core logic)
  - `utils.py` (metrics & plots)
  - `main.py` (Streamlit interface)

---

## 📸 Sample Gallery

![Demo Screenshot 1](/images/projects/digital-image-processing-lib/demo1.png)  
![Demo Screenshot 2](/images/projects/digital-image-processing-lib/demo2.png)  
![Demo Screenshot 3](/images/projects/digital-image-processing-lib/demo3.png)

---

## 🧠 Why It Matters

- **Educational**: Helps students learn through visual experimentation.
- **Research Utility**: Benchmark filters, compare transformations, or prep data.
- **Usability-first**: Web-based, real-time, no installation required.

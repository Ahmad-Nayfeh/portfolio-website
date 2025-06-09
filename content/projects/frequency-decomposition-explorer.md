---
title: "Frequency Decomposition Explorer"
slug: "frequency-decomposition-explorer"
date: "2024-04-10"
coverImage: "/images/projects/frequency-decomposition-explorer.jpg"
tags: ["Signal Processing", "Fourier Transform", "Frequency Analysis", "Image Processing", "Educational Tools"]
excerpt: "An interactive Streamlit app to visualize, isolate, and reconstruct 1D or 2D signals based on their frequency bands using FFT and IFFT."
category: "Signal & Image Processing"
githubLink: "https://github.com/nahmad2000/Frequency-Decomposition-Explorer"
liveDemoUrl: "https://frequency-decomposition-explorer-nahmad.streamlit.app/"
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
featured: false
---

# Frequency Decomposition Explorer

## Overview

This interactive Streamlit application enables real-time exploration of frequency components in both 1D signals and 2D images using Fast Fourier Transform (FFT). Users can upload data, define custom frequency bands, isolate specific frequency content, and reconstruct the signal/image using only selected components. The tool is designed for educational analysis, research prototyping, and visual intuition building.

## Motivation

Frequency decomposition is fundamental in signal processing, but rarely visualized interactively. This project addresses that gap by combining FFT analysis with user-controllable reconstruction, making it easier to understand how low, mid, and high frequency components shape raw signals and images. It also simulates the effect of frequency-based compression and denoising.

## Technical Approach

- **Frontend**: Built with `Streamlit` for easy deployment and interactive UI
- **FFT Core**: Uses NumPy's `fft`, `fft2`, and corresponding inverse transforms
- **Modular System**:
  - `frequency_analyzer.py`: FFT computation, band masking, filtering
  - `main.py`: Streamlit app logic and session-based UI
  - `plotter.py`: Matplotlib-based visualizations (time domain, spectrum, difference)
- **Features**:
  - 1D and 2D data support (`.npy`, `.png`, `.jpg`, etc.)
  - Customizable frequency bands (up to 8 by default)
  - Selective reconstruction using user-selected bands
  - Band-wise visualizations + difference map
  - Compression-like metrics: % of FFT coefficients retained

## Key Features / Contributions

- Real-time frequency band control with checkboxes and sliders
- FFT-based masking and band separation using radial or linear filtering
- Side-by-side comparison of original vs. reconstructed signals/images
- Difference plot to visualize removed frequency content
- Easily extendable architecture (ideal for applying to denoising or compression pipelines)

## Results & Findings

- Demonstrated clear trade-offs between frequency selection and signal fidelity
- Achieved meaningful data reduction (up to ~90% coefficient removal) with visually minimal distortion in low-pass reconstructions
- Exposed noise and sharp edge contributions by isolating high-frequency bands


![Streamlit UI Demo](/images/projects/frequency-decomposition-explorer/demo1.png)

## Reflection

This project sharpened my understanding of frequency-domain transformations and practical filtering techniques. It also allowed me to apply signal theory in a hands-on way, combining interactive UI design with numerical algorithms. Going forward, I plan to extend it to support wavelet transforms and integrate benchmark datasets (like ECG signals, DICOM images) for more real-world testing.
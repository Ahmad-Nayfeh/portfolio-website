---
title: "DIP-Lib: Interactive Digital Image Processing Toolkit"
slug: "digital-image-processing-lib"
date: "2025-04-30"
coverImage: "/images/projects/digital-image-processing-lib.jpg"
tags: ["Image Processing", "Computer Vision"]
excerpt: "A Streamlit app for learning classical image processing: nine modules with live parameters, stacked into a pipeline that shows the image after every step."
category: "Lab"
githubLink: "https://github.com/Ahmad-Nayfeh/Digital-Image-Processing-Library"
liveDemoUrl: "https://digital-image-processing-library-nahmad.streamlit.app/"
featured: false
highlights:
  - "Nine classical image processing modules"
  - "Stack steps into a pipeline and see each stage"
  - "OpenCV and scikit-image behind a Streamlit UI"
---

# DIP-Lib

> A place to see what an image processing operation does by moving a slider, instead of reading about it.

[Open the app](https://digital-image-processing-library-nahmad.streamlit.app/)

## How it works

Upload an image, pick a module, adjust its parameters, and preview the effect. Add it to the pipeline, repeat, then run the pipeline: the app shows the image after every step, reports PSNR and SSIM between the original and the result, and lets you download it. Steps can be removed and the pipeline cleared at any time.

## The nine modules

- **Downsampling and interpolation:** shrink and enlarge with different methods (nearest, bilinear, Lanczos).
- **Geometric transformations:** rotate, scale, translate and shear.
- **Noise:** add Gaussian or salt-and-pepper noise, then remove it with Gaussian blur, median or non-local means filters.
- **Enhancement:** gamma correction, histogram equalisation and CLAHE.
- **Lighting correction:** blur subtraction in the spatial domain, or homomorphic filtering in the frequency domain.
- **Edge detection:** Sobel, Scharr, Laplacian and Canny.
- **Sharpening:** unsharp masking.
- **Thresholding:** global or adaptive.
- **Colour spaces:** convert between BGR, RGB, HSV and LAB, and view each channel.

## Screens

![Uploading an image and choosing a module](/images/projects/digital-image-processing-lib/demo1.png)
![Previewing an enhancement with before-and-after colour histograms](/images/projects/digital-image-processing-lib/demo2.png)
![A two-step pipeline with the image after each step and final PSNR and SSIM](/images/projects/digital-image-processing-lib/demo3.png)

## Built with

Python, OpenCV, NumPy and scikit-image for the processing, Streamlit for the interface.

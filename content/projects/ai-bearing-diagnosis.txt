---
title: "AI-Powered Bearing Fault Diagnosis"
slug: "ai-bearing-diagnosis"
date: "2025-06-09"
coverImage: "/images/projects/ai-bearing-diagnosis.jpg"
tags: ["Machine Learning", "Deep Learning", "Signal Processing", "Predictive Maintenance", "1D CNN", "Streamlit"]
excerpt: "An end-to-end deep learning project that uses a 1D Convolutional Neural Network (CNN) to diagnose bearing faults from vibration signals, deployed as an interactive Streamlit web app."
category: "Predictive Maintenance"
githubLink: "https://github.com/ahmad-nayfeh/ai-bearing-diagnosis"
liveDemoUrl: "https://ai-bearing-diagnosis-nahmad.streamlit.app/"
challenge: "A key challenge was handling the inherent class imbalance in the CWRU bearing dataset, where healthy signals are far more common than faulty ones. Another challenge was designing an inference system that could process real-world signals of any length, as the CNN model requires a fixed-size input."
solution: "To address class imbalance, I used the Synthetic Minority Over-sampling Technique (SMOTE) to generate new data points for the underrepresented fault classes, creating a balanced training set. For handling variable-length signals in the Streamlit app, I implemented a robust preprocessing pipeline that pads short signals and uses a sliding window approach for long signals, ensuring consistent and accurate predictions."
technologies:
  - "Python"
  - "PyTorch"
  - "Streamlit"
  - "Scikit-learn"
  - "Pandas & NumPy"
  - "Imbalanced-learn"
  - "Matplotlib & Seaborn"
features:
  - "Fault diagnosis for Normal, Inner Race, Outer Race, and Ball faults"
  - "Interactive web interface for real-time signal diagnosis"
  - "Robustly handles signals of any length (padding/windowing)"
  - "1D CNN model optimized for vibration signal data"
  - "Data augmentation using SMOTE to handle class imbalance"
  - "Performance comparison between models trained on balanced vs. imbalanced data"
featured: true
---

# AI-Powered Bearing Fault Diagnosis

## Overview

This project provides an end-to-end solution for automated bearing fault diagnosis using deep learning. It leverages a 1D Convolutional Neural Network (CNN) to analyze raw vibration signals from machinery and accurately classify the health of the bearing. The entire workflow, from data exploration and preprocessing to model training and deployment, is encapsulated in a series of Jupyter notebooks and a final, user-friendly Streamlit web application.

## The Challenge

In predictive maintenance, a primary goal is to detect failures before they happen. However, real-world sensor data presents significant challenges. The collected data is often imbalanced, with a surplus of 'normal' operation signals and a scarcity of 'fault' signals. Furthermore, a model trained on fixed-length segments needs a way to handle continuous, variable-length data streams during live inference. This project tackles both of these core issues head-on.

## Technical Approach

The solution is broken down into a clear, multi-step process:

- **Data Analysis (EDA)**: The project begins with a thorough exploratory data analysis of the public Case Western Reserve University (CWRU) bearing dataset to understand its characteristics.
- **Preprocessing & Augmentation**: Raw signals are segmented into windows suitable for the CNN. To combat data imbalance, the training set is balanced using the SMOTE algorithm from the `imbalanced-learn` library.
- **1D CNN Model Development**: A custom 1D CNN architecture is built using PyTorch, specifically designed to learn discriminative features from time-series vibration data. The model is trained and evaluated on both the original and the SMOTE-balanced datasets to demonstrate the performance improvement.
- **Interactive Deployment**: The trained model and a `scikit-learn` scaler are deployed in a Streamlit application. This app features an intelligent prediction function that can process uploaded signal files of any length by applying padding or a sliding-window technique before feeding them to the model.

## Key Features

- **Real-Time Diagnosis**: Users can upload a CSV file containing a vibration signal and receive an instant diagnosis of the bearing's condition.
- **Robust Signal Handling**: The inference logic is designed to be flexible, correctly analyzing signals that are shorter or longer than the model's expected input size.
- **High-Performance Model**: The 1D CNN architecture is effective at capturing the complex patterns in vibration data, leading to accurate fault classification.
- **Demonstrated Impact of SMOTE**: The project clearly shows that training on a balanced dataset significantly improves the model's ability to correctly identify less common fault types.

## Results

The trained 1D CNN model demonstrates high accuracy in classifying the four different bearing states. The experiments documented in the notebooks show that the model trained on the SMOTE-balanced dataset outperforms the one trained on the original, imbalanced data, particularly in its recall for the minority fault classes.

![Streamlit UI Demo](/images/projects/ai-bearing-diagnosis/demo.png)

## Reflection

This project was a fantastic exercise in building a complete, real-world machine learning application. It highlighted the critical importance of preprocessing and data balancing—proving that the model is only as good as the data it's trained on. Implementing the 1D CNN in PyTorch was a great learning experience, and using Streamlit to deploy the final model made it incredibly easy to create an interactive and shareable demo. Future work could involve testing the model on different datasets, experimenting with more advanced deep learning architectures like LSTMs or Transformers, and optimizing the model for deployment on edge devices for true real-time monitoring.
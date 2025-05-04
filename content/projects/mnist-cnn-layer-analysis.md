---
title: "MNIST CNN Layer Analysis"
slug: "mnist-cnn-layer-analysis"
date: "2024-02-28"
coverImage: "/images/projects/mnist-cnn-layer-analysis.jpg"
tags: ["Deep Learning", "Model Complexity", "Supervised Learning", "Image Processing", "Visualization"]
excerpt: "An educational deep learning experiment comparing test accuracy and model complexity across CNN architectures for handwritten digit classification."
category: "Deep Learning"
githubLink: "https://github.com/nahmad2000/MNIST-CNN-Layer-Analysis"
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

# MNIST CNN Layer Analysis

## Overview

This project investigates how adding convolutional layers to a neural network affects both accuracy and complexity for handwritten digit classification using the MNIST dataset. Conducted as part of a Digital Image Processing course, the work compares multiple architectures from a simple fully connected model to deeper CNNs, examining the trade-offs between test performance and parameter count.

## Motivation

In deep learning, increasing model depth can enhance performance—but often at the cost of computational complexity. This project was designed to provide an intuitive, visual analysis of how convolutional layers influence model expressiveness, accuracy, and efficiency when applied to a classic classification problem.

## Technical Approach

- **Dataset**: MNIST (60,000 training, 10,000 test images of size 28×28 grayscale)
- **Preprocessing**: Normalization, one-hot encoding, reshaping to include channel dimension
- **Architectures Compared**:
  - `Baseline_1FC`: Fully connected (Dense) layer only
  - `Model_1Conv`: One Conv2D + MaxPooling2D block
  - `Model_2Conv`: Two Conv2D + MaxPooling2D blocks
- **Training Setup**:
  - Optimizer: Adam
  - Loss: Categorical cross-entropy
  - Epochs: 5
  - Batch size: 32
- **Evaluation**:
  - Accuracy on test set
  - Total trainable parameters
  - Scatter plot of Accuracy vs. Parameter Count (log scale)

## Key Features / Contributions

- Designed a minimal yet structured experiment to isolate the impact of architectural depth
- Implemented and evaluated three progressively complex models under identical training conditions
- Produced clear visualizations and tabulated metrics to guide architectural insights
- Interpreted results in terms of performance trade-offs and parameter efficiency

## Results & Findings

| Model         | Test Accuracy | Trainable Parameters |
|---------------|----------------|----------------------|
| Baseline_1FC  | 0.9780         | 101,770              |
| Model_1Conv   | 0.9853         | 693,962              |
| Model_2Conv   | 0.9893         | 225,034              |

- The **first Conv layer** significantly improved accuracy, though with a sharp increase in parameter count.
- The **second Conv layer** provided even better accuracy while **reducing parameters**—thanks to pooling reducing the spatial resolution before the dense layers.

![Accuracy vs Parameters Plot](/images/projects/mnist-cnn-layer-analysis/accuracy_vs_params.png)

## Reflection

This project clarified how deeper CNNs can improve performance while potentially reducing over-parameterization through better architectural design. It strengthened my understanding of convolutional operations, parameter budgeting, and trade-off analysis in model development—key skills in building efficient deep learning systems.


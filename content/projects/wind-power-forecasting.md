---
title: "Wind Power Forecasting"
slug: "wind-power-forecasting"
date: "2024-03-15"
coverImage: "/images/projects/wind-power-forecasting.jpg"
tags: ["Time Series Forecasting", "Deep Learning", "Supervised Learning", "AI", "Model Complexity"]
excerpt: "Hybrid deep learning and machine learning pipeline for wind power forecasting using direct LSTM and LSTM→Random Forest architectures."
category: "AI for Energy"
githubLink: "https://github.com/nahmad2000/WindPowerForecasting-LSTM-RF"
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

# Wind Power Forecasting

## Overview

This project tackles the challenge of short-term wind power prediction using a hybrid approach that combines deep learning and ensemble methods. The pipeline implements two forecasting strategies: (1) **Direct LSTM** that predicts power output from raw inputs, and (2) **Indirect LSTM→Random Forest**, where LSTM predicts meteorological features used by a Random Forest to estimate power. The solution includes robust data preprocessing, model training, visual analytics, and performance benchmarking.

## Motivation

Accurate wind power forecasting is critical for grid stability and energy planning. This project simulates a real-world scenario of turbine-level forecasting using publicly available data and explores whether hybrid learning models can improve accuracy, robustness, or interpretability over pure deep learning models.

## Technical Approach

- **Dataset**: Hourly wind turbine data (`Active Power`, `Wind Speed`, `Wind Direction`, `Power Curve`)
- **Preprocessing**:
  - Resampling to hourly intervals
  - Handling missing values with forward/backward filling
  - Scaling with `MinMaxScaler`
- **Two Forecasting Pipelines**:
  - **Direct**: LSTM → Power
  - **Indirect**: LSTM → Features → Random Forest → Power
- **Modeling**:
  - LSTM: 2-layer Keras Sequential model with early stopping & learning curves
  - RF: 50-tree scikit-learn model using predicted LSTM features
- **Evaluation Metrics**: MAE, RMSE, R², IA, SDE, MAPE
- **Visualization**:
  - Actual vs. Predicted curves
  - Feature distributions and ACF/PACF
  - Correlation heatmaps and learning curves

## Key Features / Contributions

- End-to-end pipeline for sequential time series forecasting
- Fully modular: config-driven experimentation across approaches
- Resilient evaluation framework with inverse scaling and data windowing
- Automated saving of models, plots, metrics for both pipelines
- Rich set of reusable utilities for scaling, splitting, and sequence generation

## Results & Findings

| Approach | MAE | RMSE | R² | IA | SDE | MAPE |
|----------|-----|------|-----|-----|------|------|
| **Direct LSTM** | 231.81 | 375.24 | 0.9261 | 0.9805 | 374.72 | 288.02 |
| **Indirect (LSTM → RF)** | 429.12 | 720.31 | 0.7278 | 0.9170 | 711.10 | 1017.82 |

- **Direct LSTM** significantly outperforms the hybrid pipeline in all metrics, especially error-based ones (RMSE, MAPE)
- The hybrid method's reduced performance suggests feature drift between predicted features and ground truth inputs for RF


#### Output Examples: Learning Curves 

| Direct LSTM                                                             | Indirect LSTM                                                               |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ![Direct LSTM Learning](/images/projects/wind-power-forecasting/lstm_direct_learning_curves.png) | ![Indirect LSTM Learning](/images/projects/wind-power-forecasting/lstm_indirect_learning_curves.png) |
|                                                                         |                                                                             |

#### Output Examples:: Actual vs Predicted (Test Set)

| Direct LSTM                                                                            | Indirect RF                                                                        |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| ![Direct Actual vs Predicted](/images/projects/wind-power-forecasting/lstm_direct_actual_vs_predicted_test.png) | ![RF Actual vs Predicted](/images/projects/wind-power-forecasting/rf_indirect_actual_vs_predicted_test.png) |


> 📌 **Insight**: The Direct LSTM approach outperforms the Indirect method in all evaluation metrics on the test set, particularly in RMSE and MAPE.

## Reflection

This project deepened my understanding of temporal forecasting, especially the effects of model architecture on downstream performance. It also offered hands-on practice with pipeline design, modularization, and interpretable evaluation in time-series modeling. Potential next steps include testing exogenous variables, adding weather forecast integration, or transitioning to probabilistic models.


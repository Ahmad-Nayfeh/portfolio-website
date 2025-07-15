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
featured: false
---

<style>
.project-container-simple {
font-family: 'Inter', sans-serif;
line-height: 1.75;
}
.project-header-simple {
text-align: center;
padding: 2rem 0;
border-bottom: 1px solid hsl(var(--border));
margin-bottom: 2.5rem;
}
.project-header-simple h1 {
font-size: 2.5rem;
font-weight: 800;
letter-spacing: -0.025em;
margin-bottom: 0.5rem;
}
.project-header-simple p {
font-size: 1.125rem;
color: hsl(var(--muted-foreground));
max-width: 700px;
margin: 0 auto;
}
.project-section-simple {
margin-bottom: 3rem;
}
.project-section-simple h2 {
font-size: 1.75rem;
font-weight: 700;
color: hsl(var(--primary));
margin-bottom: 1rem;
padding-bottom: 0.5rem;
border-bottom: 2px solid hsl(var(--primary) / 0.1);
}
.project-section-simple p, .project-section-simple ul {
color: hsl(var(--foreground) / 0.9);
}
.project-section-simple ul {
list-style-position: outside;
padding-left: 1.5rem;
}
.project-section-simple li {
margin-bottom: 0.5rem;
}
.video-link-container {
text-align: center;
margin: 2rem 0;
}
.video-link-button {
display: inline-flex;
align-items: center;
gap: 0.5rem;
background-color: #FF0000;
color: white;
padding: 0.75rem 1.5rem;
border-radius: 0.5rem;
text-decoration: none;
font-weight: 600;
transition: background-color 0.3s ease, transform 0.2s ease;
}
.video-link-button:hover {
background-color: #CC0000;
transform: translateY(-2px);
}
.image-container-simple {
margin: 2rem 0;
text-align: center;
background: hsl(var(--muted) / 0.4);
padding: 1rem;
border-radius: 0.75rem;
border: 1px solid hsl(var(--border));
}
.image-container-simple img {
border-radius: 0.5rem;
box-shadow: 0 4px 15px rgba(0,0,0,0.1);
max-width: 100%;
}
.image-container-simple figcaption {
margin-top: 0.75rem;
font-size: 0.9rem;
color: hsl(var(--muted-foreground));
font-style: italic;
}
</style>

<div class="project-container-simple">
<div class="project-header-simple">
<h1>AI-Powered Bearing Fault Diagnosis</h1>
<p>An end-to-end deep learning project that uses a 1D Convolutional Neural Network (CNN) to diagnose bearing faults from vibration signals, deployed as an interactive Streamlit web app.</p>
</div>

<div class="video-link-container">
    <a href="https://youtu.be/F31jicRn_A8" target="_blank" rel="noopener noreferrer" class="video-link-button">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-youtube"><path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17"/><path d="m10 15 5-3-5-3z"/></svg>
        <span>Watch the Video Demo</span>
    </a>
</div>

<div class="project-section-simple">
    <h2>Overview</h2>
    <p>This project provides an end-to-end solution for automated bearing fault diagnosis using deep learning. It leverages a 1D Convolutional Neural Network (CNN) to analyze raw vibration signals from machinery and accurately classify the health of the bearing. The entire workflow, from data exploration and preprocessing to model training and deployment, is encapsulated in a series of Jupyter notebooks and a final, user-friendly Streamlit web application.</p>
</div>

<div class="project-section-simple">
    <h2>The Challenge</h2>
    <p>In predictive maintenance, a primary goal is to detect failures before they happen. However, real-world sensor data presents significant challenges. The collected data is often imbalanced, with a surplus of 'normal' operation signals and a scarcity of 'fault' signals. Furthermore, a model trained on fixed-length segments needs a way to handle continuous, variable-length data streams during live inference. This project tackles both of these core issues head-on.</p>
</div>

<div class="project-section-simple">
    <h2>Technical Approach</h2>
    <p>The solution is broken down into a clear, multi-step process:</p>
    <ul>
        <li><strong>Data Analysis (EDA)</strong>: The project begins with a thorough exploratory data analysis of the public Case Western Reserve University (CWRU) bearing dataset to understand its characteristics.</li>
        <li><strong>Preprocessing & Augmentation</strong>: Raw signals are segmented into windows suitable for the CNN. To combat data imbalance, the training set is balanced using the SMOTE algorithm from the `imbalanced-learn` library.</li>
        <li><strong>1D CNN Model Development</strong>: A custom 1D CNN architecture is built using PyTorch, specifically designed to learn discriminative features from time-series vibration data. The model is trained and evaluated on both the original and the SMOTE-balanced datasets to demonstrate the performance improvement.</li>
        <li><strong>Interactive Deployment</strong>: The trained model and a `scikit-learn` scaler are deployed in a Streamlit application. This app features an intelligent prediction function that can process uploaded signal files of any length by applying padding or a sliding-window technique before feeding them to the model.</li>
    </ul>
</div>

<div class="project-section-simple">
    <h2>Key Features</h2>
    <ul>
        <li><strong>Real-Time Diagnosis</strong>: Users can upload a CSV file containing a vibration signal and receive an instant diagnosis of the bearing's condition.</li>
        <li><strong>Robust Signal Handling</strong>: The inference logic is designed to be flexible, correctly analyzing signals that are shorter or longer than the model's expected input size.</li>
        <li><strong>High-Performance Model</strong>: The 1D CNN architecture is effective at capturing the complex patterns in vibration data, leading to accurate fault classification.</li>
        <li><strong>Demonstrated Impact of SMOTE</strong>: The project clearly shows that training on a balanced dataset significantly improves the model's ability to correctly identify less common fault types.</li>
    </ul>
</div>

<div class="project-section-simple">
    <h2>Results</h2>
    <p>The trained 1D CNN model demonstrates high accuracy in classifying the four different bearing states. The experiments documented in the notebooks show that the model trained on the SMOTE-balanced dataset outperforms the one trained on the original, imbalanced data, particularly in its recall for the minority fault classes.</p>
    <div class="image-container-simple">
        <figure>
            <img src="/images/projects/ai-bearing-diagnosis/demo.png" alt="Streamlit UI Demo" />
            <figcaption>The interactive Streamlit UI for real-time fault diagnosis.</figcaption>
        </figure>
    </div>
</div>

<div class="project-section-simple">
    <h2>Reflection</h2>
    <p>This project was a fantastic exercise in building a complete, real-world machine learning application. It highlighted the critical importance of preprocessing and data balancing—proving that the model is only as good as the data it's trained on. Implementing the 1D CNN in PyTorch was a great learning experience, and using Streamlit to deploy the final model made it incredibly easy to create an interactive and shareable demo. Future work could involve testing the model on different datasets, experimenting with more advanced deep learning architectures like LSTMs or Transformers, and optimizing the model for deployment on edge devices for true real-time monitoring.</p>
</div>

</div>
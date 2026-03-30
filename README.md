# GPR-Diffusion: Diffusion Neural Networks for Dynamic Process Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation of a **diffusion neural network** for modeling dynamic processes, developed as part of a bachelor's thesis at Bauman Moscow State Technical University. The model learns to generate realistic temporal trajectories by reversing a gradual noising process, enabling data generation with distributions similar to the original dynamics.

<img width="1929" height="1971" alt="prediction on 60" src="https://github.com/user-attachments/assets/ba8a9680-8fb7-4c5a-bbb0-d7d82a51b5a5" />
*Orange: diffusion prediction, Blue: used data, Green: real state. The diffusion model generates data with a distribution similar to the original.*

## ✨ Features

- 🧠 **Diffusion-based generative model** for sequential/dynamic data
- 📈 **Gaussian Process Regression (GPR)** integration for uncertainty estimation
- 🔄 Support for training on custom temporal datasets
- 📊 Visualization tools for comparing real vs generated trajectories
- ⚙️ Prefect workflows for experiment orchestration

## 📁 Project Structure
gpr-diffusion/
├── data/ # Dataset handling and preprocessing
├── experiments/ # Experiment configurations and notebooks
├── models/ # Diffusion model architecture (UNet, noise schedulers)
├── prefect_experiments/ # Prefect flows for reproducible experiments
├── tools/ # Utility functions (visualization, metrics)
├── requirements.txt # Python dependencies
└── README.md

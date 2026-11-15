# 🎫 Customer Support Ticket Routing using GRU Networks

<p align="center">

  **Intelligent Customer Support Ticket Classification & Attention-Based Priority Ranking**

  <br/>

  ![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
  ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-purple)
  ![GRU](https://img.shields.io/badge/Deep%20Learning-GRU-green)
  ![Status](https://img.shields.io/badge/Status-Completed-success)

</p>

---

## 📌 Project Overview

**Customer Support Ticket Routing** is an intelligent Natural Language Processing (NLP) system designed to automate customer support ticket classification, routing, and priority ranking.

The system analyzes customer support messages and automatically predicts the appropriate support category and priority level using Machine Learning and Deep Learning techniques.

It combines traditional Machine Learning algorithms with GRU-based neural networks and an attention mechanism to improve automated ticket understanding and classification.

The project implements a GRU-based architecture for ticket routing and priority prediction, helping support teams manage incoming requests more efficiently.

## 🎯 Project Objectives

- Automate customer support ticket classification.
- Route tickets to appropriate support categories.
- Predict ticket priority using deep learning.
- Apply NLP techniques to understand customer messages.
- Implement GRU networks for sequential text processing.
- Integrate an attention mechanism for improved feature representation.
- Compare traditional Machine Learning models with GRU-based models.
- Improve customer support workflow efficiency.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎫 Ticket Classification | Automatically classifies customer support tickets |
| 🔀 Automated Routing | Predicts the appropriate support category |
| ⚡ Priority Prediction | Identifies ticket priority |
| 🧠 GRU Networks | Uses Gated Recurrent Units for text classification |
| 🎯 Attention Mechanism | Focuses on important information in ticket text |
| 📊 ML Model Comparison | Compares multiple traditional ML algorithms |
| 🔍 NLP Processing | Processes and transforms customer messages |
| 📈 Prediction Analysis | Supports classification and priority evaluation |
| 🖥️ Interactive Interface | Provides a user interface for ticket analysis |

## 🏗️ System Architecture

The system follows a Natural Language Processing and Deep Learning pipeline.

```text
       ┌───────────────────────────┐
       │   Customer Support Ticket │
       │       Input Text          │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │     Text Preprocessing    │
       │  Cleaning & Tokenization  │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │       Tokenization        │
       │     Vocabulary Mapping    │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │       GRU Networks        │
       │   Sequential Processing  │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │    Attention Mechanism    │
       │ Important Feature Focus   │
       └─────────────┬─────────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
    ┌────────────────┐  ┌────────────────┐
    │ Ticket Routing│  │Priority Ranking│
    │    Model       │  │     Model      │
    └───────┬────────┘  └───────┬────────┘
            │                  │
            ▼                  ▼
    ┌────────────────┐  ┌────────────────┐
    │ Support Queue  │  │ Priority Level │
    └────────────────┘  └────────────────┘
```

## 🧠 Models Implemented

### 1. Traditional Machine Learning Models

The project evaluates multiple traditional machine learning algorithms for customer support ticket classification.

- **Naive Bayes (NB):** Probabilistic text classification.
- **Logistic Regression (LR):** Linear classification approach.
- **Support Vector Machine (SVM):** Margin-based classification.
- **Random Forest (RF):** Ensemble-based classification.

### 2. GRU-Based Deep Learning

The project uses Gated Recurrent Unit (GRU) networks to process sequential text information.

GRU helps capture contextual dependencies in customer support messages while learning meaningful representations of ticket text.

### 3. Attention-Based Priority Ranking

An attention mechanism is integrated with the GRU architecture to focus on important information within customer messages.

The system uses separate trained GRU models for:

- Ticket routing.
- Ticket priority prediction.

## ⚙️ Dataset Information

The project uses the dataset:

`dataset-tickets-multi-lang-4-20k.csv`

| Dataset Attribute | Value |
|---|---

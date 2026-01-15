# Stock Market Prediction via Financial News Sentiment & Reject Option Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NLP](https://img.shields.io/badge/NLP-FinBERT-green)
![ML](https://img.shields.io/badge/Model-XGBoost%20%7C%20LSTM-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project implements a machine learning pipeline for predicting Apple Inc. (AAPL) stock movements. The approach integrates domain-specific Natural Language Processing (NLP) with a **Reject Option Learning (ROL)** mechanism.

The primary objective is to enhance risk-adjusted returns by combining quantitative technical indicators with qualitative sentiment signals extracted from financial news, while allowing the model to abstain from prediction during periods of high uncertainty.

## 📌 Methodology

### 1. Financial Sentiment Analysis (FinBERT)
To process unstructured textual data, we utilized **FinBERT**, a BERT model pre-trained specifically on financial corpora.
*   **Dataset:** The pipeline processes over **29,000 news articles** from 2016 to 2024.
*   **Feature Extraction:** Unlike generalist models, FinBERT provides domain-aware sentiment classification. We converted the model's output into continuous sentiment scores to be used as features for the downstream classifiers.

### 2. Strategic Abstention (ROL & GARCH)
Standard classification models enforce a binary prediction (Long/Short) for every data point. We implemented a **Reject Option** layer to mitigate false positives.
*   **Confidence Thresholding:** The model calculates a confidence score for every prediction. If confidence is below a threshold ($\tau$), the trade is rejected.
*   **Volatility Filter:** We integrated a **GARCH(1,1)** model to estimate conditional volatility. During regimes of high market volatility, the rejection threshold is dynamically adjusted to preserve capital.

## 📊 Key Results
Comparing the baseline classifier (Standard XGBoost) against the proposed method (XGBoost + ROL + GARCH):

| Metric | Standard XGBoost | **Proposed Method (With ROL)** |
| :--- | :---: | :---: |
| **Sharpe Ratio** | 3.83 | **5.84** |
| **Accuracy** | 65.0% | **66.4%** (on accepted trades) |
| **Coverage** | 100% | **82.1%** |
| **Max Drawdown** | -99.9% | **-98.8%** |

*Note: The reduction in coverage (17.9% of trades rejected) corresponds to periods of low model confidence or high volatility.*

## 🛠️ Tech Stack
*   **Data Sources:** Kaggle (News Corpus), Yahoo Finance (Market Data).
*   **NLP:** `transformers` (Hugging Face), `torch`.
*   **Modeling:** `xgboost`, `tensorflow` (Keras for LSTM), `arch` (for GARCH models).
*   **Analysis:** `scikit-learn`, `pandas`, `shap`.

## 📉 Dataset & Setup
Due to GitHub's file size limits, the datasets are hosted externally.

### 📥 [DOWNLOAD DATASETS HERE (Google Drive)]()

**Instructions:**
1.  **Clone the repository.**
2.  **Download the files** from the Drive link:
    *   `apple_news_data.csv`: Raw news dataset.
    *   `df_with_indicators_classification.csv`: Pre-processed dataset containing technical indicators and FinBERT sentiment scores.
3.  **Place the files** in the root directory of the project.
4.  **Run the notebook**.

> **Note:** The NLP processing step (FinBERT) is computationally expensive. To reproduce the ML results immediately, load the pre-processed `df_with_indicators_classification.csv` file directly in the notebook.

## 🖼️ Visualizations

### Reject Option Implementation
*Blue lines indicate accepted trades; Red lines indicate rejection due to uncertainty.*
![Price Series](paper/rol_price_series.png)

### ROL Trade-off Analysis
*Relationship between Confidence Threshold ($\tau$) and Conditional Sharpe Ratio.*
![ROL Tradeoff](paper/rol_tradeoff.png)

## 📄 Repository Structure
*   `notebooks/`: Main Jupyter Notebook containing the full pipeline.
*   `paper/`: Final academic report (PDF).
*   `requirements.txt`: Python dependencies.

## 👥 Authors
*   **Pietro Tommaso Giannattasio**
*   **Francesco Mosca**

*Machine Learning Course Project - Sapienza University of Rome (2025)*

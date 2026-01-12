# AAPL Stock Prediction via Financial News Sentiment & Machine Learning

This repository contains the final project for the Machine Learning course. The project explores the integration of qualitative financial news sentiment with quantitative technical indicators to predict the market dynamics of Apple Inc. (AAPL).

## 🚀 Project Overview
Predicting stock movements is challenging due to market noise and non-stationarity. This project implements a dual-stage pipeline:
1. **Sentiment Analysis:** We used **FinBERT** (a transformer model pre-trained on financial corpora) to quantify the sentiment of nearly 30,000 news articles spanning from 2016 to 2024.
2. **Predictive Modeling:** We compared a Deep Learning architecture (**Hybrid CNN-BiLSTM**) for price jump classification and an Ensemble model (**XGBoost**) for daily return regression.
3. **Interpretability:** We applied **SHAP (SHapley Additive exPlanations)** to demystify the "black box" of the models and identify the key drivers of price movements.

## 📊 Key Results
- **Regression (XGBoost):** Achieved an **$R^2$ score of 0.81**, explaining over 80% of the return variance.
- **Classification (Bi-LSTM):** Achieved an overall **Accuracy of 79%** and an **AUC of 0.74**.
- **Insights:** SHAP analysis revealed that Bollinger Band positions and EMA differences are the primary predictors, with news sentiment acting as a robust secondary confirmation signal.

## 📂 Repository Structure
*   `notebooks/`: Contains the main `.ipynb` file with the full pipeline (preprocessing, sentiment extraction, training, and evaluation).
*   `requirements.txt`: List of Python libraries required to run the project.
*   `.gitignore`: Configuration to prevent large data files and temporary files from being uploaded.

## 📉 Dataset Information
The news dataset used in this project is the **"Apple Stock (AAPL): Historical Financial News Data"** from Kaggle.
- **Size:** ~100 MB (29,752 entries).
- **Source:** [Download on Kaggle](https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data).

**Note:** Due to GitHub's file size limits, the raw CSV dataset is not included in this repository. To run the code:
1. Download the dataset from the Kaggle link above.
2. Place the `apple_news_data.csv` file inside the directory where you run the notebook.

**Note on Running the Code:**
The preprocessing section includes a Sentiment Analysis pipeline using FinBERT, which processes ~30k articles and may take several hours to complete. To facilitate the evaluation, I have provided the pre-processed sentiment scores and the final merged dataset (df_con_indicatori.csv) in the data/ folder. You can skip the "Sentiment Analysis" section and load the provided CSVs directly to run the Machine Learning models (LSTM and XGBoost) immediately.


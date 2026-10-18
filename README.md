# Twitter Sentiment Analysis with NLP

An end-to-end Natural Language Processing project for classifying the sentiment of tweets using the Sentiment140 dataset.

The project applies text preprocessing and machine learning techniques to predict whether a tweet expresses a negative or positive sentiment.

## Overview

Social media contains a large amount of public opinion data. This project explores how Natural Language Processing can be used to transform raw tweets into useful sentiment insights.

The goal is to build and evaluate a sentiment classifier capable of identifying the emotional polarity of a tweet.

## Features

- Text cleaning and normalization
- Tokenization and stopword removal
- Exploratory Data Analysis (EDA)
- Feature extraction with TF-IDF
- Sentiment classification with machine learning models
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Interactive prediction for custom tweets

## Dataset

This project uses the [Sentiment140 dataset](http://help.sentiment140.com/for-students), a large collection of tweets labeled by sentiment.

The original labels are:

| Label | Sentiment |
|------:|-----------|
| 0 | Negative |
| 4 | Positive |

For this project, the labels are converted into a binary classification format:

- `0`: Negative
- `1`: Positive

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
twitter-sentiment-analysis-nlp/
├── data/
│   └── sentiment140.csv
├── notebooks/
│   └── sentiment_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── sentiment_classifier.pkl
├── requirements.txt
└── README.md

Workflow
1. Load and explore the Sentiment140 dataset.
2. Clean tweet text by removing links, mentions, punctuation, and special characters.
3. Normalize text using lowercase conversion, tokenization, and stopword removal.
4. Transform the processed text into numerical features using TF-IDF.
5. Train classification models.
6. Evaluate model performance using classification metrics.
7. Save the best model for future predictions.
Models
The project compares machine learning models such as:
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine
The best-performing model is selected based on evaluation metrics and validation results.
Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
Final metrics will be added after the training pipeline is completed and validated.

Example Prediction
predict_sentiment("I love how easy this application is to use!")
Expected output:
Positive
Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/twitter-sentiment-analysis-nlp.git
cd twitter-sentiment-analysis-nlp
Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate
Install the dependencies:
pip install -r requirements.txt
Run the notebook:
jupyter notebook
Future Improvements
- Deploy an interactive interface using Streamlit
- Add neutral sentiment classification
- Test transformer-based models such as BERT
- Create a REST API for real-time predictions
- Integrate sentiment monitoring dashboards

Author
Samuel Macedo
LinkedIn · GitHub
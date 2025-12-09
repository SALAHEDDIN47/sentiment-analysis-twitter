# 🐦 Twitter Sentiment Analysis

A comprehensive real-time Twitter sentiment analysis system using machine learning and natural language processing.

## 🚀 Features

- **Real-time Twitter data collection** using Tweepy
- **Advanced NLP preprocessing** with spaCy and NLTK
- **Sentiment analysis** using TwhIN-BERT model
- **PostgreSQL database** for data storage
- **Interactive dashboard** with Streamlit
- **Multi-language support** (English focused)

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Twitter Developer Account

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis

## ⚙️ Configuration

1. **Create virtual environment**
python -m venv venv

source twitter_sentiment_env/bin/activate  # Linux/Mac
# or
twitter_sentiment_env\Scripts\activate  # Windows

2. **Install dependencies**
pip install -r requirements.txt


3. **Setup environment variables**
cp .env.example .env
# Edit .env with your credentials

4. **Initialize database**
python setup_database.py

## database schema

CREATE TABLE tweets (
    id SERIAL PRIMARY KEY,
    tweet_id BIGINT UNIQUE,
    text TEXT,
    clean_text TEXT,
    created_at TIMESTAMP,
    user_name VARCHAR(255),
    sentiment_label VARCHAR(50),
    sentiment_score FLOAT,
    confidence FLOAT,
    language VARCHAR(10),
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

## import the dataset from kaggle

https://www.kaggle.com/datasets/kazanova/sentiment140
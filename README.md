# Emotion-Journal-Analyzer
Streamlit-based NLP dashboard for emotion and mental health analysis from journal entries
# 💖 Emotion Journal Analyzer

A Streamlit-based NLP dashboard that analyzes emotions, sentiment, and mental health stability from large-scale journal text data.

## 🚀 Features
- Emotion detection using NRC Lexicon
- Sentiment analysis (Polarity & Subjectivity)
- Mental Health Stability Score
- DistilBERT-based emotion classification
- Mood clustering using K-Means
- Interactive dashboards with Plotly & Seaborn
- CSV upload and analyzed data download

## 🧠 Tech Stack
- Python
- Streamlit
- NLP (SpaCy, NRCLex, TextBlob)
- Transformers (DistilBERT)
- Machine Learning (Scikit-learn)
- Data Visualization (Plotly, Seaborn)

## 📂 Input Format
CSV file with at least:
- `text` column (journal entries)
- Optional: `Timestamp`, `status`

## ▶️ How to Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py

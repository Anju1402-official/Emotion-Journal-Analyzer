import streamlit as st
import pandas as pd
import plotly.express as px
from nrclex import NRCLex
from textblob import TextBlob
import spacy
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Emotion Journal Analyzer", layout="wide")
# bump upload limit to ~1 GB

# --- PASSWORD PROTECTION ---
def login():
    st.sidebar.subheader("🔐 Login Required")
    password = st.sidebar.text_input("Enter password", type="password")
    if password == "anju123":
        return True
    elif password:
        st.sidebar.warning("❌ Incorrect password. Try again.")
    return False

if not login():
    st.stop()

st.title("💖 Emotion Journal Analyzer Dashboard")

# upload
uploaded_file = st.file_uploader("Upload your CSV file with at least 1000 journal entries", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV file with a 'text' column to begin analysis. Minimum 1000 entries recommended.")
    st.stop()

# load
df = pd.read_csv(uploaded_file)
st.sidebar.success(f"✅ Loaded {len(df)} entries")  # confirm all rows are in memory

if 'text' not in df.columns:
    st.error("Uploaded file must contain a 'text' column.")
    st.stop()

# Load models
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- CACHED FUNCTIONS ---
@st.cache_data(show_spinner=False)
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

@st.cache_data(show_spinner=False)
def get_emotions(text: str) -> str:
    emo = NRCLex(text)
    return ", ".join([e for e, _ in emo.top_emotions])

@st.cache_data(show_spinner=False)
def get_sentiment(text: str) -> tuple[float, float]:
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# preprocess & drop nulls
df = df.dropna(subset=['text'])
df['Preprocessed Text'] = df['text'].apply(preprocess)

# emotion detection & sentiment
with st.spinner("Analyzing emotions and sentiments..."):
    # NRC
    df['Detected Emotions'] = df['Preprocessed Text'].apply(get_emotions)

    # batch DistilBERT
    texts = df['Preprocessed Text'].tolist()
    ml_results = classifier(texts, batch_size=32)
    df['ML Emotion'] = [r['label'] for r in ml_results]

    # batch TextBlob
    sentiments = [get_sentiment(t) for t in df['text']]
    df[['Sentiment Polarity', 'Subjectivity Score']] = pd.DataFrame(
        sentiments, columns=['Sentiment Polarity', 'Subjectivity Score']
    )

# stability score
def stability_score(polarity, subjectivity):
    return round((1 - abs(polarity)) * (1 - subjectivity) * 100, 2)

df['Mental Health Stability Score'] = df.apply(
    lambda row: stability_score(row['Sentiment Polarity'], row['Subjectivity Score']),
    axis=1
)

# --- DASHBOARD METRICS ---
st.subheader("📊 Emotion & Sentiment Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", len(df))
col2.metric("Avg. Polarity", round(df['Sentiment Polarity'].mean(), 2))
col3.metric("Avg. Stability Score", round(df['Mental Health Stability Score'].mean(), 2))

# correlation heatmap
st.subheader("🧪 Correlation Heatmap")
corr = df[['Sentiment Polarity', 'Subjectivity Score', 'Mental Health Stability Score']].corr()
fig_corr, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# pie chart
st.subheader("🥧 Emotion Distribution")
emotion_counts = df['ML Emotion'].value_counts().reset_index()
emotion_counts.columns = ['Emotion', 'Count']
fig_pie = px.pie(emotion_counts, values='Count', names='Emotion', title='Detected Emotion Distribution')
st.plotly_chart(fig_pie, use_container_width=True)

# box plot
st.subheader("📦 Stability Score Box Plot by Emotion")
fig_box = px.box(df, x='ML Emotion', y='Mental Health Stability Score', points="all")
st.plotly_chart(fig_box, use_container_width=True)

# clustering
if len(df) >= 3:
    st.subheader("🔄 Mood Trajectory Clustering")
    X = df[['Sentiment Polarity', 'Subjectivity Score']].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    df['Mood Cluster'] = kmeans.labels_
    fig_cluster = px.scatter(
        df, x='Sentiment Polarity', y='Subjectivity Score',
        color='Mood Cluster', hover_data=['text'],
        title="K-Means Clustering of Mood Signals"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# model eval
if 'status' in df.columns:
    st.subheader("📊 Model Evaluation")
    report = classification_report(df['status'], df['ML Emotion'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    cm = confusion_matrix(df['status'], df['ML Emotion'])
    labels = sorted(df['status'].unique())
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

# timeline
st.subheader("📈 Emotion Over Time")
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    fig_time = px.scatter(
        df, x='Timestamp', y='Sentiment Polarity', color='ML Emotion',
        title="Emotion Timeline 🕒", 
        labels={'Sentiment Polarity': 'Polarity'},
        hover_data=['text']
    )
    st.plotly_chart(fig_time, use_container_width=True)

# data table & download
st.subheader("🎨 Detailed Data Table")
st.dataframe(df, use_container_width=True)

st.download_button(
    "📅 Download Analyzed Data",
    data=df.to_csv(index=False),
    file_name="analyzed_emotions.csv",
    mime="text/csv"
)
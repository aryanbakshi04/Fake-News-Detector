# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. PAGE CONFIG
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# 2. ADVANCED CLEANING FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

# 3. LOAD AND PREP DATA
@st.cache_data
def load_data():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1
    df_all = pd.concat([df_fake, df_true], ignore_index=True)
    df_all = df_all.drop(columns=["title", "subject", "date"], errors='ignore')
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    df_all["text"] = df_all["text"].apply(clean_text)
    return df_all

df = load_data()

# 4. SPLIT DATA AND TRAIN MODELS
X = df["text"]
y = df["class"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vect, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vect, y_train)

# 5. STREAMLIT UI
st.title("📰 Fake News Detector")

st.markdown("Enter a news article below to classify it using ensemble predictions from Logistic Regression and Random Forest.")

input_text = st.text_area("Paste news article here...", height=200)

if st.button("Classify"):
    if not input_text.strip():
        st.warning("Please enter news content.")
    else:
        cleaned = clean_text(input_text)
        vec = vectorizer.transform([cleaned])
        pred_lr = lr_model.predict(vec)[0]
        pred_rf = rf_model.predict(vec)[0]

        if pred_lr == pred_rf:
            result = "True News" if pred_lr == 1 else "Fake News"
        else:
            result = "Not Sure"

        st.markdown("---")
        st.subheader("Prediction Results")
        st.write(f"**Logistic Regression:** {'True News' if pred_lr else 'Fake News'}")
        st.write(f"**Random Forest:** {'True News' if pred_rf else 'Fake News'}")

        st.markdown("### Final Decision:")
        if result == "True News":
            st.success("TRUE NEWS")
        elif result == "Fake News":
            st.error("FAKE NEWS")
        else:
            st.info("NOT SURE")

st.markdown("---\nBuilt with scikit-learn, pandas, Streamlit.")

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1) Page config must be first
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# 2) Define your cleaning function once
def clean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\n', ' ', text)
    return text.strip()

# 3) Cache data loading & preprocessing
@st.cache_data(show_spinner=False)
def load_and_prepare():
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1

    # drop last 10 as optional holdout, keep the rest for train
    df_train = pd.concat([df_fake.iloc[:-10], df_true.iloc[:-10]], ignore_index=True)
    df_train = df_train.drop(columns=["title","subject","date"])
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train["text"] = df_train["text"].apply(clean)
    return df_train

# 4) Cache model training (resource to persist across reruns)
@st.cache_resource(show_spinner=False)
def train_models():
    df = load_and_prepare()
    X = df["text"]
    y = df["class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    vect = TfidfVectorizer(max_features=5000)
    Xv = vect.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000).fit(Xv, y_train)
    dt = DecisionTreeClassifier(random_state=42).fit(Xv, y_train)
    return vect, lr, dt

vectorizer, lr, dt = train_models()

# —— UI ——  
st.title("📰 Fake vs. True News Classifier")

st.markdown(
    """
    Paste any news article or snippet below and click **Classify**.  
    - Both say **True** → **True News**  
    - Both say **Fake** → **Fake News**  
    - Otherwise → **Not Sure**  
    """
)

user_input = st.text_area("Enter the news text here", height=200)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Classifying..."):
            txt = clean(user_input)
            x_vec = vectorizer.transform([txt])
            pred_lr = lr.predict(x_vec)[0]
            pred_dt = dt.predict(x_vec)[0]

            label = {0: "Fake News", 1: "True News"}
            out_lr = label[pred_lr]
            out_dt = label[pred_dt]

            if pred_lr == pred_dt:
                final = out_lr
            else:
                final = "Not Sure"

        st.subheader("Model Predictions")
        st.write(f"**Logistic Regression**: {out_lr}")
        st.write(f"**Decision Tree**: {out_dt}")
        st.markdown("---")
        st.subheader("Final Decision")
        if final == "True News":
            st.success(final)
        elif final == "Fake News":
            st.error(final)
        else:
            st.info(final)

st.markdown(
    """
    ---
    **Model Details**  
    • Logistic Regression trained on TF–IDF (5 k vocab)  
    • Decision Tree (default settings)  
    """
)

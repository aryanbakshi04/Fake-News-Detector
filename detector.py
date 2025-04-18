# app.py
import streamlit as st
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)
import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# —— Data loading & preprocessing ——
@st.cache_data
def load_and_prepare():
    # load
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1
    # hold out last 10 of each as test (optional)
    df_train = pd.concat([df_fake.iloc[:-10], df_true.iloc[:-10]], ignore_index=True)
    # drop unused columns
    df_train = df_train.drop(columns=["title","subject","date"])
    # shuffle
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    # simple clean function
    def clean(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\n', ' ', text)
        return text.strip()
    df_train["text"] = df_train["text"].apply(clean)
    return df_train

df = load_and_prepare()

# split
X_train, X_hold, y_train, y_hold = train_test_split(df["text"], df["class"], test_size=0.2, random_state=42)

# vectorize
vectorizer = TfidfVectorizer(max_features=5000)
Xv_train = vectorizer.fit_transform(X_train)

# train models
lr = LogisticRegression(max_iter=1000)
lr.fit(Xv_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(Xv_train, y_train)

# —— Streamlit UI ——
# st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake vs. True News Classifier")

st.markdown(
    """
    Paste any news article or snippet below and click **Classify**.
    We use a Logistic Regression and a Decision Tree, then:
    - If both agree **True**, we say **True News**  
    - If both agree **Fake**, we say **Fake News**  
    - Otherwise **Not Sure**  
    """
)

user_input = st.text_area("Enter the news text here", height=200)

if st.button("Classify"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        # clean & vectorize
        def clean(text):
            text = text.lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>+', '', text)
            text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub(r'\w*\d\w*', '', text)
            text = re.sub(r'\n', ' ', text)
            return text.strip()
        txt = clean(user_input)
        x_vec = vectorizer.transform([txt])
        pred_lr = lr.predict(x_vec)[0]
        pred_dt = dt.predict(x_vec)[0]

        # map to labels
        label_map = {0: "Fake News", 1: "True News"}
        out_lr = label_map[pred_lr]
        out_dt = label_map[pred_dt]

        # ensemble logic
        if pred_lr == pred_dt:
            final = out_lr
        else:
            final = "Not Sure"

        # display
        st.subheader("Model Predictions")
        st.write(f"**Logistic Regression**: {out_lr}")
        st.write(f"**Decision Tree**: {out_dt}")
        st.markdown("---")
        st.subheader(" Final Decision")
        if final == "True News":
            st.success(" True News")
        elif final == "Fake News":
            st.error(" Fake News")
        else:
            st.info(" Not Sure")

st.markdown(
    """
    ---
    **Model Details**  
    • Logistic Regression trained on TF–IDF features (5 k vocab)  
    • Decision Tree with default settings  
    """
)

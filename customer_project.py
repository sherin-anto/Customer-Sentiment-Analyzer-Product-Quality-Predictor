import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from streamlit_lottie import st_lottie
import requests
nltk.download('stopwords')

# Load Lottie animation from URL
def load_lottieurl(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

# Load and preprocess the dataset
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df = df.dropna(subset=['Review', 'Rate'])
    df = df[pd.to_numeric(df['Rate'], errors='coerce').notna()]
    df['Rate'] = df['Rate'].astype(str).str.extract(r'(\d)').astype(int)
    df['Sentiment'] = df['Rate'].apply(lambda x: 'Positive' if x >= 4 else 'Neutral' if x == 3 else 'Negative')

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
        return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])

    df['CleanReview'] = df['Review'].apply(clean_text)
    return df

# Train model
def train_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['CleanReview'])
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, tfidf

# Streamlit App

def run_app(model, tfidf):
    st.set_page_config(layout="wide")
    st.title("🛒 E-commerce Review Sentiment Analyzer")

    # Analyze Single Review
    st.header("🔍 Analyze a Single Review")
    review = st.text_area("Enter a product review:")

    if st.button("Analyze", key="analyze_single") and review.strip():
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        def clean_input(text):
            text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
            return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])

        cleaned = clean_input(review)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 'Positive':
            st.success("✅ Positive Review")
            lottie_pos = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_4kx2q32n.json")
            if lottie_pos:
                st_lottie(lottie_pos, speed=1, height=400)
            else:
                st.info("✅ Positive sentiment animation unavailable.")

        elif result == 'Negative':
            st.error("❌ Negative Review")
            lottie_neg = load_lottieurl("https://lottie.host/f731e8e8-cdbf-4d76-b3cb-9dfe4015cfa2/mwDEobzTN7.json")
            if lottie_neg:
                st_lottie(lottie_neg, speed=1, height=400)
            else:
                st.info("❌ Negative sentiment animation unavailable.")

        else:
            st.warning("⚠ Neutral Review")
            lottie_neu = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_pNx6yH.json")
            if lottie_neu:
                st_lottie(lottie_neu, speed=1, height=400)
            else:
                st.info("😐 Neutral sentiment animation unavailable.")

    elif st.button("Analyze", key="analyze_single_empty"):
        st.warning("⚠ Please enter a review to analyze.")

    # Upload CSV
    st.header("📂 Upload a CSV for Bulk Review Analysis")
    uploaded_file = st.file_uploader("Upload CSV with a column named 'Review'", type=["csv"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            if 'Review' in df_uploaded.columns:
                df_uploaded = df_uploaded.dropna(subset=['Review'])
                stemmer = PorterStemmer()
                stop_words = set(stopwords.words('english'))
                def clean_uploaded(text):
                    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
                    return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])
                df_uploaded['CleanReview'] = df_uploaded['Review'].apply(clean_uploaded)
                tfidf_vectors = tfidf.transform(df_uploaded['CleanReview'])
                df_uploaded['Predicted Sentiment'] = model.predict(tfidf_vectors)
                st.success("✅ Sentiment analysis completed!")
                st.dataframe(df_uploaded[['Review', 'Predicted Sentiment']])
                csv = df_uploaded.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", csv, "sentiment_results.csv", "text/csv")
            else:
                st.error("❌ The uploaded file must contain a 'Review' column.")
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")

# Main
if __name__ == '__main__':
    try:
        data = load_and_prepare_data("d:/genlab project/flipkart_product.csv")
        model, tfidf = train_model(data)
        run_app(model, tfidf)
    except Exception as e:
        st.error(f"❌ Failed to start app: {e}")

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
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')

def load_lottieurl(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

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

    # Convert price if available
    if 'Mentioned_Price' in df.columns:
        df['Price_Clean'] = df['Mentioned_Price'].astype(str).str.replace("₹", "").str.replace(",", "").astype(float)
        bins = [0, 5000, 10000, 20000, 50000, float('inf')]
        labels = ['< ₹5K', '₹5K–₹10K', '₹10K–₹20K', '₹20K–₹50K', '> ₹50K']
        df['Price_Range'] = pd.cut(df['Price_Clean'], bins=bins, labels=labels)

    return df

def train_model(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['CleanReview'])
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, tfidf, X_test, y_test

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        color: #008080;
        font-weight: 800;
        text-align: center;
        margin-bottom: 25px;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 10px;
    }
    .sub-header {
        font-size: 24px;
        color: #1f77b4;
        font-weight: bold;
        margin-top: 20px;
        background-color: #e0f7fa;
        padding: 8px;
        border-left: 6px solid #00acc1;
        border-radius: 4px;
    }
    .highlight-box {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #8bc34a;
    }
    .footer-note {
        text-align: center;
        font-size: 14px;
        color: #aaa;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛒 E-commerce Review Sentiment Analyzer</div>", unsafe_allow_html=True)

file_path = "d:/genlab project/flipkart_product.csv"
data = load_and_prepare_data(file_path)
model, tfidf, X_test, y_test = train_model(data)

st.markdown("<div class='sub-header'>Sentiment Summary</div>", unsafe_allow_html=True)
pos = (data['Sentiment'] == 'Positive').sum()
neg = (data['Sentiment'] == 'Negative').sum()
neu = (data['Sentiment'] == 'Neutral').sum()
col1, col2, col3 = st.columns(3)
col1.metric("✅ Positive", pos, f"{round(pos/len(data)*100,1)}%")
col2.metric("❌ Negative", neg, f"{round(neg/len(data)*100,1)}%")
col3.metric("😐 Neutral", neu, f"{round(neu/len(data)*100,1)}%")

st.markdown("<div class='sub-header'>📊 Rating Distribution</div>", unsafe_allow_html=True)
fig, ax = plt.subplots()
data['Rate'].value_counts().sort_index().plot(kind='bar', color='coral', ax=ax)
ax.set_xlabel("Rating (1 to 5 stars)")
ax.set_ylabel("Count")
st.pyplot(fig)

st.markdown("<div class='sub-header'>📈 Sentiment Distribution</div>", unsafe_allow_html=True)
fig, ax = plt.subplots()
data['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
ax.axis('equal')
st.pyplot(fig)

st.markdown("<div class='sub-header'>🌥 Word Clouds</div>", unsafe_allow_html=True)
for sentiment in ['Positive', 'Negative', 'Neutral']:
    st.markdown(f"**{sentiment}**")
    text = " ".join(data[data['Sentiment'] == sentiment]['CleanReview'])
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

if 'Mentioned_Damage' in data.columns:
    st.markdown("<div class='sub-header'>📦 Mentioned Damage Distribution</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    data['Mentioned_Damage'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

if 'Delivery_Time' in data.columns:
    st.markdown("<div class='sub-header'>🚚 Delivery Time Frequency</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    data['Delivery_Time'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

if 'Price_Range' in data.columns:
    st.markdown("<div class='sub-header'>💰 Sentiment by Price Range</div>", unsafe_allow_html=True)
    sentiment_by_price = data.groupby(['Price_Range', 'Sentiment']).size().unstack().fillna(0)
    st.bar_chart(sentiment_by_price)

st.markdown("<div class='sub-header'>📊 Model Evaluation</div>", unsafe_allow_html=True)
y_pred = model.predict(X_test)
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df.style.highlight_max(axis=0))

st.markdown("<div class='sub-header'>🔍 Analyze a Single Review</div>", unsafe_allow_html=True)
with st.form("single_review_form"):
    review = st.text_area("💬 Enter your product review:")
    submitted = st.form_submit_button("✨ Analyze")

if submitted and review.strip():
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def clean_input(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
        return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])

    cleaned = clean_input(review)
    vector = tfidf.transform([cleaned])
    result = model.predict(vector)[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        if result == 'Positive':
            st.success("✅ Positive Review")
            st.markdown("### ⭐⭐⭐⭐⭐")
            lottie = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_4kx2q32n.json")
        elif result == 'Negative':
            st.error("❌ Negative Review")
            st.markdown("### ⭐⭐")
            lottie = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")
        else:
            st.warning("😐 Neutral Review")
            st.markdown("### ⭐⭐⭐")
            lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_pNx6yH.json")
    with col2:
        if lottie:
            st_lottie(lottie, height=300)
        else:
            st.info("⚠️ Animation unavailable.")

st.markdown("<div class='sub-header'>📂 Bulk Review Sentiment Analyzer</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV with a column named 'Review'", type=["csv"])
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        if 'Review' in df_uploaded.columns:
            df_uploaded = df_uploaded.dropna(subset=['Review'])

            def clean_uploaded(text):
                text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
                return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])

            df_uploaded['CleanReview'] = df_uploaded['Review'].apply(clean_uploaded)
            tfidf_vectors = tfidf.transform(df_uploaded['CleanReview'])
            df_uploaded['Predicted Sentiment'] = model.predict(tfidf_vectors)
            st.success("✅ Sentiment analysis completed!")
            st.dataframe(df_uploaded[['Review', 'Predicted Sentiment']])
            csv = df_uploaded.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "sentiment_results.csv", "text/csv")
        else:
            st.error("❌ The uploaded file must contain a 'Review' column.")
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {e}")

st.markdown("<div class='footer-note'>Made with ❤️ for GenLab Data Science Project</div>", unsafe_allow_html=True)

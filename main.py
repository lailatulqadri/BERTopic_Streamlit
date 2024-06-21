import streamlit as st
import pandas as pd
from bertopic import BERTopic

# Fungsi untuk membaca file CSV
def load_data(file):
    data = pd.read_csv(file)
    return data

# Fungsi untuk menjalankan BERTopic
def generate_topics(data, n_gram_range=(1, 1)):
    # Asumsi bahwa data memiliki kolom 'text'
    texts = data['text'].tolist()
    model = BERTopic(n_gram_range=n_gram_range)
    topics, _ = model.fit_transform(texts)
    return topics

# Streamlit UI
st.title('BERTopic with Streamlit')

# Input teks dari user
user_input = st.text_area("Enter text:")

if user_input:
    model = BERTopic()
    topics, _ = model.fit_transform([user_input])
    st.write(f"Topic: {topics[0]}")

# Input file CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    st.write(data.head())

    topics = generate_topics(data)
    data['topic'] = topics
    st.write(data)

# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Load your data
@st.cache_data
def load_data():
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Perform BERTopic modeling
@st.cache_resource
def create_model(data):
    # Asumsi bahwa data memiliki kolom 'text'
    texts = data['text'].tolist()
    model = BERTopic(n_gram_range=n_gram_range,calculate_probabilities=True)
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
    st.write("Sample data:")
    st.write(data.head())

    topics = generate_topics(data)
    data['topic'] = topics
    st.write("Data with topics:")
    st.write(data)

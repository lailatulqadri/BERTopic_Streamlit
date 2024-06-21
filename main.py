# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic

# Load your data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Perform BERTopic modeling
#@st.cache_resource
def create_model(data, n_gram_range=(1, 2)):
    texts = data['text'].tolist()
    model = BERTopic(n_gram_range=n_gram_range, calculate_probabilities=True)
    topics, _ = model.fit_transform(texts)
    return topics

# Generate topics for the data
def generate_topics(data):
    n_gram_range = (1, 1)  # Adjust the n-gram range if necessary
    topics = create_model(data, n_gram_range)
    return topics

# Streamlit UI
st.title('BERTopic with Streamlit')

# Input text from user
user_input = st.text_area("Enter text:")

if user_input:
    model = BERTopic()
    topics, _ = model.fit_transform([user_input])
    st.write(f"Topic: {topics[0]}")

# Input file CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("Sample data:")
        st.write(data.head())

        topics = generate_topics(data)
        data['topic'] = topics
        st.write("Data with topics:")
        st.write(data)

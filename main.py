# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
import joblib



# Load your data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Perform BERTopic modeling
#@st.cache_resource
def create_model(data,num_topics, min_topic_size, nr_topics):
    texts = data['text'].tolist()
    n_gram_range=(1,3)
    model = BERTopic(n_gram_range,calculate_probabilities=True,nr_topics=num_topics, min_topic_size=min_topic_size, nr_top_words=nr_topics)
    topics, _ = model.fit_transform(texts)
    st.write(model.get_topic_info())
    fig = model.visualize_topics()
    # Display the figure in Streamlit
    st.plotly_chart(fig)
    joblib.dump(model, 'bertopic_model.pkl')
    return topics

# Generate topics for the data
#def generate_topics(data):
#    n_gram_range = (1, 3)  # Adjust the n-gram range if necessary
#    topics = create_model(data, n_gram_range)
#    return topics

# Streamlit UI
st.title('BERTopic with Streamlit')

# Input parameter to setup BERTopic
# Streamlit UI
st.write("Please provide BERTopic parameters as follows:")
# Number of Topics

num_topics = st.number_input("Number of Topics", min_value=2, max_value=50, value=5)
# Min Topic Size
min_topic_size = st.number_input("Minimum Topic Size", min_value=2, max_value=50, value=5)
# Nr of Top Words
nr_topics = st.number_input("Number of Top Words", min_value=2, max_value=30, value=5)
st.write('Note:')
st.info("Number of Topic : defines the number of topics BERTopic should try to extract from the corpus", icon="ℹ️")

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

        topics = create_model(data)
        data['topic'] = topics
        st.write("Data with topics:")
        st.write(data)
       

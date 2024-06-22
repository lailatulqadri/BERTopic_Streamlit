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
def create_model(data, n_gram_range=(1, 3)):
    texts = data['text'].tolist()
    model = BERTopic(n_gram_range=n_gram_range, calculate_probabilities=True)
    topics, _ = model.fit_transform(texts)
    st.write(model.get_topic_info())
    fig = model.visualize_topics()
    # Display the figure in Streamlit
    st.plotly_chart(fig)
    joblib.dump(model, 'bertopic_model.pkl')
    return topics

# Generate topics for the data
def generate_topics(data):
    n_gram_range = (1, 3)  # Adjust the n-gram range if necessary
    topics = create_model(data, n_gram_range)
    return topics

# Streamlit UI
st.title('BERTopic with Streamlit')

# Input parameter to setup BERTopic
# Streamlit UI
st.write("Please provide BERTopic parameters as follows:")
# Number of Topics

num_topics = st.number_input("Number of Topics", min_value=2, max_value=50, value=10)
# Min Topic Size
min_topic_size = st.number_input("Minimum Topic Size", min_value=2, max_value=50, value=10)
# Nr of Top Words
nr_topics = st.number_input("Number of Top Words", min_value=2, max_value=30, value=10)
st.write('Note:')
st.info('This is a purely informational message', icon="ℹ️")
st.write('Number of Topic: defines the number of topics that BERTopic should try to extract from the corpus. If set to a specific number, the model will attempt to generate exactly that many topics.')
st.write('Minimum Topic Size: specifies the minimum number of documents that should be present in a topic. Topics with fewer documents than this threshold will be merged with other topics.')
st.write('Number of Top Words: provide description here')

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
       

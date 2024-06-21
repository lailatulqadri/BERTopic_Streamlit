# Import necessary libraries
import streamlit as st
import numpy as np
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Load your data
@st.cache_data
def load_data():
    data = fetch_20newsgroups(subset='all')['data']
    return np.array(data)  # Convert list to numpy array

# Perform BERTopic modeling
@st.cache_data
def create_model(data):
    model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    topics, probs = model.fit_transform(data.tolist())  # Convert numpy array back to list
    return model, topics, probs

# Streamlit app
def main():
    st.title("BERTopic with Streamlit")
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Loading data...done!')

    model, topics, probs = create_model(data)

    st.write('Number of topics:', model.get_topic_freq().shape[0] - 1)

    topic_to_visualize = st.slider('Select topic to visualize', min_value=0, max_value=model.get_topic_freq().shape[0] - 2, value=0, step=1)
    st.write(model.visualize_distribution(probs[topic_to_visualize]))

if __name__ == "__main__":
    main()

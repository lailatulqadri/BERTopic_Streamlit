# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Load your data
#@st.cache_data
def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data['text'].values.tolist()  # Replace 'column_name' with the name of the column that contains the text data
    return None

# Perform BERTopic modeling
@st.cache_resource
def create_model(data):
    if data is not None:
        model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
        topics, probs = model.fit_transform(data)
        return model, topics, probs
    return None, None, None

def main():
    st.title("BERTopic with Streamlit")
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Loading data...done!')

    if data is not None:
        model, topics, probs = create_model(data)
        st.write(model)
        

        if model is not None:
            st.write('Number of topics:', model.get_topic_freq().shape[0] - 1)

            #topic_to_visualize = st.slider('Select topic to visualize', min_value=0, max_value=model.get_topic_freq().shape[0] - 2, value=0, step=1)
            #st.write(model.visualize_distribution(probs[topic_to_visualize]))
            #document_index = st.slider('Select document to visualize', min_value=0, max_value=len(probs) - 1, value=0, step=1)
            document_index = 10
            st.write(model.visualize_distribution(probs[document_index]))
        else:
            st.write("Please upload a CSV file.")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()

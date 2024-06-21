import streamlit as st
import pandas as pd
from bertopic import BERTopic
import numpy as np

def bertopic_wrapper(data):
    """
    Performs BERTopic topic modeling on the provided text data.

    Args:
        data (pd.Series or list): A pandas Series containing text data or a list of text strings.

    Returns:
        BERTopic model: The trained BERTopic model.
        pd.DataFrame: A DataFrame containing the generated topics and their probabilities for each document.
    """

    # Preprocess data (remove unnecessary steps, handle potential errors)
    data = pd.Series(data)  # Ensure data is a Series

    # Create BERTopic model instance (adjust parameters as needed)
    model = BERTopic(language="english", top_n_words=10, min_topic_size=3)

    # Fit the model on the text data
    topics, probs = model.fit_transform(data)

    # Create a DataFrame for topics and probabilities
    topic_df = pd.DataFrame({"Topic": topics, "Probability": probs.max(axis=1)})

    return model, topic_df

def main():
    """
    Streamlit app for BERTopic topic modeling with user-uploaded CSV data.
    """

    st.title("BERTopic Topic Modeling with Streamlit")

    # Upload CSV data
    uploaded_file = st.file_uploader("Upload your CSV data:", type="csv")

    if uploaded_file is not None:
        try:
            # Read CSV data into a DataFrame
            df = pd.read_csv(uploaded_file)

            # Select text column (handle potential errors)
            if "text" not in df.columns:
                st.error("Please ensure your CSV has a column named 'text' containing the text data.")
                return

            text_data = df["text"]

            # Perform BERTopic modeling
            model, topic_df = bertopic_wrapper(text_data)

            # Display results
            st.subheader("Topics and Probabilities")
            st.write(topic_df)

            # Optional visualizations (consider adding interactive charts)
            # st.bar_chart(topic_df["Topic"].value_counts())

            # Allow downloading topics (consider using st.download_button)
            # download_csv = st.button("Download Topics")
            # if download_csv:
            #     topic_df.to_csv("topics.csv", index=False)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

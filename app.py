import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np
import requests
import json

# Set up your GROQ API key and endpoint
API_KEY = "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"  # Replace with your actual API key
API_URL = "https://api.groq.com/v1/your_endpoint"  # Replace with the correct Groq endpoint

# Function to preprocess the dataset
def preprocess_data(data):
    data_processed = data.copy()
    encoders = {}
    
    # Encode categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    # Normalize numerical columns
    scaler = MinMaxScaler()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data_processed[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data_processed, encoders

# Function to get adjusted weights from Groq API using Llama3-8b-8192
def get_adjusted_weights_with_groq(data):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload with data for Llama3
    prompt = f"""
    Analyze the following data and suggest appropriate weights for each feature to optimize the similarity calculation. 
    Data: {data}
    Please return the suggested weights as a JSON object.
    """

    payload = {
        "model": "Llama3-8b-8192",  # Specify the Llama3 model
        "prompt": prompt,
        "temperature": 0.7,  # You can tweak this value for creativity
        "max_tokens": 100  # Adjust token limit as per your needs
    }
    
    # Send the request to the Groq API
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['text']  # Adjust this based on the API response structure
    else:
        st.error("Error from Groq API: " + response.text)
        return {}

# Weighted similarity calculation
def calculate_weighted_similarity(row1, row2, data, weights):
    similarity = 0
    total_weight = sum(weights.values())
    
    for col in data.columns:
        weight = weights.get(col, 0)
        if weight == 0:
            continue
        
        if data[col].dtype in ['int64', 'float64']:  # Numerical column
            diff = abs(row1[col] - row2[col])
            similarity += (1 - diff) * weight
        
        else:  # Categorical column
            sim = 1 - jaccard([row1[col]], [row2[col]])
            similarity += sim * weight
    
    return (similarity / total_weight) * 100

# Streamlit App
st.title("Row Similarity with Auto-Adjusted Weights")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Preprocess data
    data_processed, encoders = preprocess_data(data)
    
    # Prepare an example data structure (this should be your preprocessed data or a summary of the features)
    data_example = [{"column_name": col, "type": "object" if data[col].dtype == 'object' else "numeric"} for col in data.columns]
    
    # Get adjusted weights from the Llama3 model using Groq API
    adjusted_weights = get_adjusted_weights_with_groq(data_example)
    
    # Convert the result to a dictionary (assuming the result is JSON-like)
    weights = json.loads(adjusted_weights)
    
    # Define column weights (use the adjusted weights)
    st.sidebar.header("Adjust Column Weights")
    for col in data.columns:
        weights[col] = st.sidebar.slider(f"Weight for {col}", 0, 10, int(weights.get(col, 1)))
    
    # Select rows for comparison
    st.subheader("Select Rows for Comparison")
    row1_index = st.selectbox("Select Row 1", data.index)
    row2_index = st.selectbox("Select Row 2", data.index)
    
    if row1_index != row2_index:
        # Calculate similarity
        row1 = data_processed.iloc[row1_index]
        row2 = data_processed.iloc[row2_index]
        similarity = calculate_weighted_similarity(row1, row2, data_processed, weights)
        
        # Display results
        st.subheader("Similarity Results")
        st.write(f"Similarity between Row {row1_index} and Row {row2_index}: **{similarity:.2f}%**")
        
        # Display row details
        st.subheader("Row Details")
        st.write("Row 1:", data.iloc[row1_index])
        st.write("Row 2:", data.iloc[row2_index])
    else:
        st.warning("Please select two different rows for comparison.")

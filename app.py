import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np
import requests
import json

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

# Function to call Groq API (with API key from sidebar)
def call_groq_api(api_key, model_input):
    api_url = "https://api.groq.com/v1/query"  # Change this to the correct endpoint for Llama3-8b-8192
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Construct the payload
    payload = {
        "model": "llama3-8b-8192",  # Ensure this is the correct model name
        "input": model_input
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to get response from API"}

# Streamlit App
st.title("Row Similarity App")

# Sidebar for API Key input
api_key = st.sidebar.text_input("Enter your API Key", type="password")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Preprocess data
    data_processed, encoders = preprocess_data(data)
    
    # Define column weights
    weights = {col: 1 for col in data.columns}  # Default weights
    st.sidebar.header("Adjust Column Weights")
    for col in data.columns:
        weights[col] = st.sidebar.slider(f"Weight for {col}", 0, 10, 1)
    
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

        # Send similarity data to Groq model
        model_input = {
            "similarity": similarity,
            "row1_details": data.iloc[row1_index].to_dict(),
            "row2_details": data.iloc[row2_index].to_dict(),
            "weights": weights
        }
        if api_key:
            groq_response = call_groq_api(api_key, model_input)
            st.write("Groq Model Response:", groq_response)
    else:
        st.warning("Please select two different rows for comparison.")

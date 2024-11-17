import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np
import openai_secret_manager
import requests

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

# Function to analyze and adjust weights using Llama3 API
def adjust_weights_using_llama(data):
    # Prepare the message to send to the model
    prompt = f"""
    Given the following dataset:
    {data.head().to_string()}
    Analyze the data and suggest the weights for each column that would result in the most accurate row similarity comparison.
    """

    # Groq API endpoint and credentials
    api_url = "https://api.groq.com/v1/chat/completions"
    headers = {"Authorization": "Bearer YOUR_GROQ_API_KEY"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    # Send the request to the API
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        response_data = response.json()
        weights_str = response_data['choices'][0]['message']['content']
        # Convert the model's response into a dictionary of weights
        weights = {col: 1 for col in data.columns}  # Default weights
        for line in weights_str.splitlines():
            if ":" in line:
                col, weight = line.split(":")
                weights[col.strip()] = float(weight.strip())
        return weights
    else:
        st.error("Error fetching weights from Groq API")
        return {col: 1 for col in data.columns}  # Default weights if failed

# Streamlit App
st.title("Row Similarity App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Preprocess data
    data_processed, encoders = preprocess_data(data)
    
    # Automatically adjust weights using Llama3 model
    st.subheader("Adjusting Weights Using Llama3-8b-8192")
    weights = adjust_weights_using_llama(data)
    st.write("Suggested Weights:", weights)
    
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

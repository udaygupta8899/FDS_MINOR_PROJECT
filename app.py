import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import numpy as np

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

# Improved Weighted Similarity Calculation
def calculate_weighted_similarity(row1, row2, data, weights):
    similarity = 0
    total_weight = sum(weights.values())
    
    for col in data.columns:
        weight = weights.get(col, 0)
        if weight == 0:
            continue
        
        if data[col].dtype in ['int64', 'float64']:  # Numerical column
            # Relative scaling for numerical columns
            col_range = data[col].max() - data[col].min()
            if col_range > 0:  # Avoid division by zero
                diff = abs(row1[col] - row2[col]) / col_range
            else:
                diff = 0  # identical values

            similarity += (1 - diff) * weight  # Scale difference to similarity (max 1)

        else:  # Categorical column
            # Cosine similarity for categorical data, treating values as vectors
            sim = 1 - jaccard([row1[col]], [row2[col]])  # Jaccard similarity for categorical values
            similarity += sim * weight
    
    # Normalize similarity to ensure it falls between 0 and 100
    normalized_similarity = (similarity / total_weight) * 100
    return max(0, min(normalized_similarity, 100))  # Ensure the result is between 0 and 100

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
        
    else:
        st.warning("Please select two different rows for comparison.")

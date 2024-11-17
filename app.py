import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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

# Weighted similarity calculation with improved similarity measures
def calculate_weighted_similarity(row1, row2, data, weights):
    similarity = 0
    total_weight = sum(weights.values())
    
    for col in data.columns:
        weight = weights.get(col, 0)
        if weight == 0:
            continue
        
        if data[col].dtype in ['int64', 'float64']:  # Numerical column
            # Calculate the normalized difference based on the range or standard deviation
            col_range = data[col].max() - data[col].min()
            if col_range == 0:  # To avoid division by zero
                normalized_diff = 0
            else:
                normalized_diff = abs(row1[col] - row2[col]) / col_range
            similarity += (1 - normalized_diff) * weight
        
        else:  # Categorical column
            # Binary match/mismatch for categorical columns
            match = 1 if row1[col] == row2[col] else 0
            similarity += match * weight
    
    similarity = max(0, similarity)  # Ensure non-negative similarity
    return (similarity / total_weight) * 100  # Return percentage

# Streamlit App
st.title("FDS MINOR PROJECT")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Preprocess data
    data_processed, encoders = preprocess_data(data)
    
    # Automatically set weights based on column types
    weights = {}
    for col in data.columns:
        if data[col].dtype in ['object']:  # Categorical columns
            weights[col] = 3  # Assign higher weight to categorical columns
        else:  # Numerical columns
            weights[col] = 1  # Assign lower weight to numerical columns
    
    # st.sidebar.header("Column Weights (Automatic Adjustment)")
    # st.sidebar.write("Weights have been set automatically based on column types.")
    
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
        st.subheader("Person Details")
        st.write("Person 1:", data.iloc[row1_index])
        st.write("Person 2:", data.iloc[row2_index])
    else:
        st.warning("Please select two different rows for comparison.")

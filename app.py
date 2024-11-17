import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Function to preprocess the dataset
def preprocess_dataset(df):
    df_encoded = df.copy()
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df_encoded, label_encoders

# Function to calculate similarity
def calculate_similarity(row1, row2):
    # Cosine similarity for numerical vectors
    similarity = cosine_similarity([row1], [row2])
    return similarity[0][0] * 100  # Convert to percentage

# Streamlit App
st.title("FDS MINOR PROJECT")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    # Preprocess dataset
    df_encoded, label_encoders = preprocess_dataset(df)
    
    # Select rows to compare
    st.subheader("Select Rows for Comparison")
    row1_index = st.selectbox("Select Row 1", df.index)
    row2_index = st.selectbox("Select Row 2", df.index)
    
    if row1_index != row2_index:
        # Get the selected rows
        row1 = df_encoded.iloc[row1_index]
        row2 = df_encoded.iloc[row2_index]
        
        # Calculate similarity
        similarity_percentage = calculate_similarity(row1.values, row2.values)
        
        # Display similarity
        st.subheader("Comparison Results")
        st.write(f"The similarity between Row {row1_index} and Row {row2_index} is {similarity_percentage:.2f}%")
        
        # Display the differences (optional)
        st.subheader("Row Details")
        st.write("Row 1 Details:", df.iloc[row1_index])
        st.write("Row 2 Details:", df.iloc[row2_index])
    else:
        st.warning("Please select two different rows for comparison.")

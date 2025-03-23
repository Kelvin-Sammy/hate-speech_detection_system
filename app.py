 
import torch
import streamlit as st

# UI Title
st.title("Hate Speech Detection System")
st.write("Enter text below to check if it's hate speech, offensive, or neutral.")

# Input box
user_input = st.text_area("Enter Text:", "")

# Function to predict
def predict(text):
    model.eval()
    tokens = tokenizer(text, return_tensors="pt", padding='max_length', max_length=128, truncation=True).to(device)
    output = model(**tokens)
    prediction = torch.argmax(output.logits, dim=1).item()

    classes = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
    return classes[prediction]

# Predict when button is clicked
if st.button("Analyze"):
    if user_input.strip():
        result = predict(user_input)
        st.write(f"### Result: {result}")
    else:
        st.warning("Please enter text to analyze.")
    
# Import the required libraries
import streamlit as st
import requests
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Set the page configuration
st.set_page_config(
    page_title="Timelytics - Supply Chain Forecasting",
    page_icon="🕒",
    layout="wide",
)

# Display the title and app description
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - "
    "XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast "
    "Order to Delivery (OTD) times. By combining the strengths of these algorithms, Timelytics helps "
    "businesses optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays, reducing lead times "
    "and improving delivery. The model uses historical data like order processing, production, "
    "shipping times, and more to forecast OTD times for better inventory management and customer service."
)

# Cache the model loading for faster performance
@st.cache_resource
def load_model():
    url = "https://drive.google.com/file/d/1uQtb34xuPwaY4TlX43ZujiePWsjeXygs/view?usp=drive_link"
    response = requests.get(url)
    with open("voting_model.pkl", "wb") as f:
        f.write(response.content)
    with open("voting_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

voting_model = load_model()

# Prediction function
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    input_data = np.array(
        [
            [
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            ]
        ]
    )
    prediction = voting_model.predict(input_data)
    return round(prediction[0])

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    try:
        img = Image.open("./assets/supply_chain_optimisation.jpg")
        st.image(img)
    except:
        st.write("Image not found, skipping...")

    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance (in km)", value=475.35)

# Submit button
if st.button("Predict Wait Time"):
    with st.spinner("Calculating..."):
        prediction = waitime_predictor(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
        )
        st.success(f"Predicted Wait Time: **{prediction} days**")

# Sample dataset display
st.header("Sample Dataset Example")
data = {
    "Purchased Day of the Week": ["0", "3", "1"],
    "Purchased Month": ["6", "3", "1"],
    "Purchased Year": ["2018", "2017", "2018"],
    "Product Size in cm³": ["37206.0", "63714", "54816"],
    "Product Weight in grams": ["16250.0", "7249", "9600"],
    "Geolocation State Customer": ["25", "25", "25"],
    "Geolocation State Seller": ["20", "7", "20"],
    "Distance (in km)": ["247.94", "250.35", "4.915"],
}
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

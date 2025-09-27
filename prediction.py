import pickle 
import streamlit as st
import numpy as np

model = pickle.load(open('regmodel.pkl','rb'))

scaler = pickle.load(open('scaling.pkl','rb'))

st.set_page_config(page_title="🏡 House Price Prediction", layout="centered")

def mymodel():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏡 Boston House Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Enter the property details below to predict the house price.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        a1 = st.number_input('Median Income (MedInc)', min_value=0.0, format="%.2f")
        a2 = st.number_input('House Age', min_value=0.0, format="%.2f")
        a3 = st.number_input('Average Rooms', min_value=0.0, format="%.2f")
        a4 = st.number_input('Average Bedrooms', min_value=0.0, format="%.2f")
        
    with col2:
        a5 = st.number_input('Population', min_value=0.0, format="%.2f")
        a6 = st.number_input('Average Occupancy', min_value=0.0, format="%.2f")
        a7 = st.number_input('Latitude', min_value=0.0, format="%.4f")
        a8 = st.number_input('Longitude', min_value=0.0, format="%.4f")

    pred = st.button('🔍 Predict Price', use_container_width=True)

    if pred:
        input_data = np.array([[a1, a2, a3, a4, a5, a6, a7, a8]])
        
        scaled_data = scaler.transform(input_data)
        
        
        result = model.predict(scaled_data)

        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 12px; background-color: #f0f9f4; border: 1px solid #4CAF50; text-align: center;">
                <h2 style="color: #2E7D32;">✅ Predicted Price: <span style="color:#1B5E20;">{result[0]:.2f}</span></h2>
            </div>
            """,
            unsafe_allow_html=True
        )

mymodel()

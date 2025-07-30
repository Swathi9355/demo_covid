import streamlit as st
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
import google.generativeai as genai

# Class names for image model
classes = ["Covid19", "Normal"]

# Load the CNN model
ml = load_model("C:\\Users\\LENOVO\\OneDrive\\Desktop\\demo_covid\\Data\\covid_pneu_model.h5")


st.title("👩‍⚕COVID DIAGNOSIS SYSTEM🛌")

# Image upload and prediction
upl = st.file_uploader("Upload the Image", type=["png", "jpg", "jpeg"])

if upl:
    img = Image.open(upl).convert("RGB")
    st.image(img, use_column_width=True)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)

    prd = ml.predict(img_array)
    ind = np.argmax(prd[0])

    st.write("Prediction:", classes[ind])
    
    if classes[ind] == "Normal":
        st.success("Healthyyyy👍")
    else:
        st.warning("Need Diagnosis👀")  

# Dropdown and slider inputs for FastAPI model
gender_label = st.selectbox("Gender", ["Female", "Male"])
fever_label = st.selectbox("Fever", ["No", "Yes"])
cough_label = st.selectbox("Cough", ["No", "Yes"])
fatigue_label = st.selectbox("Fatigue", ["No", "Yes"])
breathlessness_label = st.selectbox("Breathlessness", ["No", "Yes"])
comorbidity_label = st.selectbox("Comorbidity", ["No", "Yes"])
stage_label = st.selectbox("Stage", ["Mild", "Moderate", "High"])
type_label = st.selectbox("Type", ["Type1", "Type2"])
tumor_size = st.slider("Tumor-size", 0, 5)
age = st.slider("Age", min_value=0, max_value=120)

# Encoding to numerical values
gender = 0 if gender_label == "Female" else 1
fever = 1 if fever_label == "Yes" else 0
cough = 1 if cough_label == "Yes" else 0
fatigue = 1 if fatigue_label == "Yes" else 0
breathlessness = 1 if breathlessness_label == "Yes" else 0
comorbidity = 1 if comorbidity_label == "Yes" else 0
stage = 0 if stage_label == "Mild" else 1 if stage_label == "Moderate" else 2
type_ = 0 if type_label == "Type1" else 1

# Data to send to FastAPI
data_input = {
    "Age": age,
    "Gender": gender,
    "Fever": fever,
    "Cough": cough,
    "Fatigue": fatigue,
    "Breathlessness": breathlessness,
    "Comorbidity": comorbidity,
    "Stage": stage,
    "Type": type_,
    "Tumor_Size": tumor_size
}

# Button to call FastAPI and display prediction
if st.button("Predict"):
    res = requests.post("http://127.0.0.1:8000/predict", json=data_input)
    response = res.json()
    st.write("Predicted Survival Rate:", response["predicted_survival_rate"])
    
    
    
    if(response["Prediction"]>0.5):
        prompt="Suggest a life style habit to overcome pneumonia or covid"
        res=chat.send_message(prompt)
        st.markdown("Suggestions")
        st.markdown(res.text)
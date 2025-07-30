import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Use a different variable name instead of 'class'
labels = ["Covid19", "Normal", "Pneumonia"]
# Load the trained model
ml = load_model(r"C:\\Users\\LENOVO\\OneDrive\\Desktop\\demo_covid\\Data\\covid_pneu_model.h5")

# Load and preprocess the image
img = image.load_img(r"C:\\Users\\LENOVO\\OneDrive\\Desktop\\demo_covid\\Data\\test\\COVID19\COVID19(467).jpg", target_size=(224, 224))
imgg = image.img_to_array(img)
imgg = np.expand_dims(imgg, axis=0) / 255.0  # Normalize and add batch dimension

# Predict
prd = ml.predict(imgg)
ind = np.argmax(prd[0])


# Output
print("Predicted class:", labels[ind])
print("Confidence scores:", prd)
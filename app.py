import os
os.system("pip install xgboost")
import xgboost


import os
os.system("pip install scikit-learn")
import sklearn

import gradio as gr


import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the XGBoost model from the pickle file
with open("xgb.pkl", "rb") as f:
    model = pickle.load(f)

def predict_rock_mine(*features):
    """
    Takes 60 sonar feature values as input, reshapes them, makes a prediction,
    and returns a string indicating whether the object is a Rock or a Mine.
    """
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    
    # Decode the prediction (assuming training labels were ['Rock', 'Mine'])
    label_encoder = LabelEncoder()
    label_encoder.fit(["Rock", "Mine"])
    result = label_encoder.inverse_transform(prediction)
    
    if result[0] == "Rock":
        return "The object is a Rock."
    else:
        return "The object is a Mine."

# Create 60 slider inputs for the sonar features (values between 0 and 1)
inputs = [gr.Slider(minimum=0.0, maximum=1.0, step=0.0001, label=f"Feature {i}") for i in range(1, 61)]

# Define the Gradio Interface
demo = gr.Interface(
    fn=predict_rock_mine,
    inputs=inputs,
    outputs="text",
    title="Rock vs Mine Prediction",
    description="Enter the 60 sonar feature values using the sliders below to predict whether the object is a Rock or a Mine."
)


demo.launch()

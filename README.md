
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

#Load the trained model
model = load_model('C:\Users\Bobby Black\Documents\AI Visual Studio Codes\AI Trainer - Accumulate H5 Model\AI Trainer - Accumulate H5 Model')

#Load and preprocess the image
img_path = 'path_to_your_image.jpg'
img = Image.open(img_path)
img = img.resize((64, 64))  # Resize the image to 64x64 (or whatever your model was trained on)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
img_array /= 255.0  # Normalize the image if that's what you did during training

#Predict
prediction = model.predict(img_array)
prediction_probability = prediction[0][0]

#New threshold
threshold = 0.75

#Interpret the prediction and print the corresponding message with probability
if prediction_probability >= threshold:
    print(f"This is an advertisement with a confidence of {prediction_probability100:.2f}%.")
else:
    print(f"This is not an advertisement with a confidence of {(1-prediction_probability)100:.2f}%.")

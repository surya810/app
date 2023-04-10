from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__, template_folder='template')

# Load the VGG16 model
with open('models/vgg16_model.pkl', 'rb') as n:
    vmodel = pickle.load(n)  
     
for layers in (vmodel.layers):
    layers.trainable = False

# Load the hybrid model
with open('models/hybrid_surya_model.pkl', 'rb') as f:
    pmodel = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess the image
     # Load and preprocess the image
    try:
        image_file = request.files['image']
        image = Image.open(image_file)
    except ValueError:
        return render_template('index.html', prediction='Error: Invalid image file')
    # image_file = request.files['image']
    # image = Image.open(image_file)

    new_size = (224, 224)
    resized_image = image.resize(new_size)
    x = img_to_array(resized_image)
    x = x/255
    # x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Extract features using the VGG16 model
    feature_extractor = vmodel.predict(x)

    # Flatten the output of the convolutional layers
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)

    # Make predictions using the hybrid model
    prediction = pmodel.predict(features)[0]

    # Return the prediction
    if prediction == 0:
        result = 'Benign'
    else:
        result = 'Malignant'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)



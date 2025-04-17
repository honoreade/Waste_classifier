import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Importing essential libraries only
import tensorflow
import numpy as np
from PIL import Image

# Loading model to backend
print("Checking backend Garbage Classifier Model")

# Declare parameters
model_Path = r"C:\Users\USER\Desktop\JupterNotebooks\models\Garbage.h5"

# Load and compile model
model = tensorflow.keras.models.load_model(model_Path, compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Labels for classification
labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def predict_image(image_path):
    """Predict waste category from image"""
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    pred = model.predict(img_array[np.newaxis, ...], verbose=0)
    predicted_class = labels[np.argmax(pred[0])]
    confidence = float(np.max(pred[0]) * 100)
    return predicted_class, confidence

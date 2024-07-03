import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Function to load the trained model
def load_trained_model(model_file='model.keras'):
    model = load_model(model_file)
    return model

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to predict the digit from an image
def predict_digit(img):
    model = load_trained_model()  # Load the trained model
    img = preprocess_image(img)  # Preprocess the image
    prediction = model.predict(img)  # Predict the digit
    predicted_class = np.argmax(prediction)  # Get the predicted class
    return predicted_class, prediction

if __name__ == "__main__":
    # Loading the MNIST test dataset
    (_, _), (X_test, y_test) = mnist.load_data()

    # Taking the first image from the test set for prediction
    img = X_test[0]

    # Predicting the digit
    predicted_class, prediction = predict_digit(img)

    # Displaying the prediction result
    print(f'Predicted Digit Class: {predicted_class}')
    print(f'Prediction Scores: {prediction}')

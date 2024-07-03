# Handwritten Digit Recognition

This project demonstrates how to build a Handwritten Digit Recognition system using TensorFlow/Keras on the MNIST dataset. It involves training a Convolutional Neural Network (CNN) to classify handwritten digits from 0 to 9.

## Project Overview

The goal of this project is to develop a machine learning model that can accurately predict the digit from an image of handwritten digits.

### Features

- **Training a CNN**: Utilize TensorFlow and Keras to construct and train a Convolutional Neural Network.
- **Dataset**: Use the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits.
- **Model Evaluation**: Evaluate the model's accuracy on a separate test set after training.
- **Prediction**: Perform real-time predictions on new handwritten digits.

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/arjunbothra/handwritten-digit-recognition-system.git
   cd handwritten-digit-recognition-system
   ```

2. **Create and Activate Virtual Environment:**

   ```bash
   # Create a virtual environment named 'env'
   python -m venv env

   # Activate the virtual environment (Windows)
   .\env\Scripts\activate

   # Activate the virtual environment (macOS/Linux)
   source env/bin/activate
   ```

3. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

4. **Run the Training Script:**

   ```bash
   python train.py
   ```

   This script will:
   - Load the MNIST dataset.
   - Preprocess the data.
   - Build and train a CNN model.
   - Save the trained model (`model.keras`) in the current directory.

5. **Evaluate Model Performance:**

   After training completes, the script will output the training history and save the model with the best validation accuracy.

## Usage

6. **Make Predictions:**

   Modify the `predict.py` script to load the trained model (`model.keras`) and perform predictions on new handwritten digits.
      ```bash
   python train.py
   ```


## File Structure

- `train.py`: Script to train the CNN model on the MNIST dataset.
- `predict.py`: Script to make predictions using the trained model.
- `model.keras`: Trained model file (generated after training).

## Credits

- [TensorFlow](https://www.tensorflow.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

# Plastic-Classification-Model

## Project Overview

**Plastic-Classification-Model** is a deep learning project that classifies images of plastic waste as either **Organic** or **Recyclable**. The project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras, and provides a user-friendly web interface using Streamlit for real-time image classification.

---

## What Was Done

### 1. Data Preparation & Exploration

- **Dataset**: Images of waste, categorized as Organic or Recyclable.
- **Preprocessing**: Images were resized, normalized, and augmented (rotation, zoom, flipping, etc.) to improve model generalization.
- **Visualization**: Data distribution and sample images were visualized to understand the dataset.

### 2. Model Building & Training

- **CNN Architecture**: Built a CNN with multiple Conv2D, MaxPooling, BatchNormalization, Dense, and Dropout layers.
- **Training**: Used data augmentation and split the data into training and validation sets.
- **Optimization**: Implemented EarlyStopping and ModelCheckpoint to avoid overfitting and save the best model.
- **Evaluation**: Trained the model for up to 30 epochs, monitored accuracy and loss, and visualized training history.

### 3. Model Evaluation

- **Testing**: Evaluated the model on a separate test set to check its generalization.
- **Metrics**: Used accuracy, classification report, and confusion matrix for performance analysis.
- **Visualization**: Displayed predictions and confusion matrix for better interpretability.

### 4. Deployment with Streamlit

- **Web App**: Developed a Streamlit app (`app.py`) that allows users to upload an image and get instant classification results.
- **User Interface**: The app displays the uploaded image, predicted class, and confidence score in a visually appealing format.

---

## How to Access and Use This Project

### 1. Clone the Repository

```sh
git clone https://github.com/iarpitsaxena/Plastic-Classification-Model.git
cd Plastic-Classification-Model
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed. Install required packages:

```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit App

Start the web application with:

```sh
streamlit run app.py
```

This will open a browser window with the Plastic Waste Classifier interface.

### 4. Use the Classifier

- Upload a clear image of plastic waste (jpg, png, jpeg).
- The app will display the image, predicted class (Organic or Recyclable), and the confidence score.

---

## Functionality of Each Library/Package

- **numpy**: Efficient numerical computations and array operations, used for image data manipulation and preprocessing.
- **pandas**: Data analysis and manipulation, especially for handling labels and tabular data.
- **matplotlib**: Visualization of data distributions, training history, and prediction results.
- **opencv-python**: Image processing tasks such as reading, resizing, and color space conversion.
- **tqdm**: Progress bars for loops, especially during data loading and preprocessing.
- **tensorflow**: Deep learning framework used to build, train, and deploy the Convolutional Neural Network (CNN) model.
- **streamlit**: Framework for building and deploying the interactive web application for image classification.
- **Pillow (PIL)**: Image loading and manipulation in the Streamlit app.

---

## Files in the Repository

- `plastic-waste-classification.ipynb`: Jupyter notebook for data exploration, model training, and evaluation.
- `app.py`: Streamlit web app for image classification.
- `best_model.keras` / `Waste-Classification-CNN-Model.h5`: Saved trained models.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

---

## Notes

- The model was trained on a dataset with two classes. For best results, use images similar to the training data.
- You can retrain the model with your own dataset by following the steps in the notebook.

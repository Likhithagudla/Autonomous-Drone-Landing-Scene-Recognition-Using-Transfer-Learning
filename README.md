# Autonomous Landing Scene Recognition Using Transfer Learning

This project implements an Autonomous Landing Scene Recognition system for drones based on Transfer Learning techniques using Convolutional Neural Networks (CNN). It leverages ResNet50 and ResNeXt50 models for scene classification and enhances accuracy by combining them with Random Forest classifiers in an ensemble model.

## OUTPUT SCREENS
![Screenshot 2025-04-08 184945](https://github.com/user-attachments/assets/753a80e3-7b45-44e5-8d49-bee4317b7c1d)

![Screenshot 2025-04-08 185000](https://github.com/user-attachments/assets/05806702-9c28-4d2b-be39-e0c90e175696)

![Screenshot 2025-04-08 185021](https://github.com/user-attachments/assets/85103c40-9601-4a54-b012-dc5cd6576c80)

![Screenshot 2025-04-08 185059](https://github.com/user-attachments/assets/0f404ca0-a1b2-40db-8657-a58fe2079331)



## Features
Dataset: Uses a dataset of aerial landing scene images categorized into 7 classes: Building, Field, Lawn, Mountain, Road, Vehicles, WaterArea, and Wilderness.

Transfer Learning: Uses ResNet50 and ResNeXt50 models pre-trained on ImageNet.

Ensemble Model: Combines ResNeXt50 with a Random Forest classifier for better scene classification performance.

Metrics Calculation: Provides accuracy, precision, recall, and F1-score for model evaluation.

Confusion Matrix: Visualizes classification performance using a confusion matrix.

Prediction: Allows prediction of landing scenes from test images using the trained models.

Graphs: Displays performance comparison between the existing ResNet50 model, proposed ResNeXt50 model, and the hybrid ensemble model.

## Requirements
Python 3.x

TensorFlow (Keras)

OpenCV

NumPy

Pandas

scikit-learn

Matplotlib

Seaborn

pickle

Keras applications (ResNet50, ResNeXt50)

## Installation
Step 1: Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/Autonomous-Landing-Scene-Recognition.git
cd Autonomous-Landing-Scene-Recognition
Step 2: Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
## Usage
Upload Dataset: Click on the "Upload Dataset" button to load your dataset containing images of landing scenes. The dataset should be organized into directories based on class labels.

Dataset Preprocessing: Click on the "Dataset Preprocessing" button to preprocess the dataset, which involves resizing images and splitting them into training and testing sets.

Run Proposed ResNeXt50 Model: Click on "Run Proposed ResNext50" to train and evaluate the ResNeXt50 model using the training data.

Run Existing ResNet50 Model: Click on "Run Existing ResNet50" to train and evaluate the ResNet50 model.

Run Extension Ensemble Random Forest: Click on "Run Extension Ensemble Random Forest" to train an ensemble model that combines the ResNeXt50 model's features with a Random Forest classifier.

Comparision Graph: Click on "Comparision Graph" to generate a bar graph comparing the performance of the three models (ResNet50, ResNeXt50, and Ensemble Random Forest).

ResNext50 Accuracy Graph: Click on "ResNext50 Accuracy Graph" to view the accuracy and loss graph of the ResNeXt50 model during training.

Predict from Test Image: Click on "Predict from Test Image" to predict the scene category for a given test image. The predicted class will be displayed along with the image.

## Model Performance
ResNet50: Pre-trained model used for transfer learning. It is trained on the ImageNet dataset and fine-tuned for the landing scene recognition task.

ResNeXt50: A more advanced transfer learning model that uses group convolutions for better feature extraction.

Ensemble Random Forest: A hybrid model that combines the feature extraction power of ResNeXt50 with the decision-making ability of a Random Forest classifier for better classification accuracy.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
The ResNet50 and ResNeXt50 models are from Keras Applications.

This project is based on transfer learning and ensemble methods for image classification.

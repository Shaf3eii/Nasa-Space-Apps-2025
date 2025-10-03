NASA Space Apps: Exoplanet Prediction AI 🪐
This project is a machine learning web application built for the NASA Space Apps Challenge. It uses data from NASA's Kepler mission to predict whether a stellar signal corresponds to a potential exoplanet.

The application consists of two main parts:

A Python script that trains a Random Forest classifier on the Kepler exoplanet dataset.

An interactive web application built with Streamlit that allows users to input signal characteristics and get a live prediction from the trained model.

!(https://www.google.com/search?q=https://i.imgur.com/YOUR_IMAGE_ID.png)
A screenshot of the running Streamlit application.

Features
High-Accuracy Model: The Random Forest model is trained on NASA's Kepler data and achieves ~93% accuracy on the test set.

Interactive Web Interface: A user-friendly web app allows for real-time predictions by adjusting sliders and input fields.

Dark Mode Theme: A sleek, modern "bare mode" interface for better usability.

Self-Contained: The project includes scripts to both train the model from scratch and run the user-facing application.

Technology Stack
Backend & Model: Python

Machine Learning: Scikit-learn

Data Manipulation: Pandas, NumPy

Web Framework: Streamlit

Model Serialization: Joblib

Project Structure
.
├── 📄 cumulative.csv      # The NASA Kepler dataset for training.
├── 🐍 train_model.py      # Script to train the ML model and save it.
├── 🖥️ app.py              # The Streamlit web application script.
├── 🧠 planet_model.joblib # The saved, pre-trained ML model.
├── ⚖️ scaler.joblib       # The saved scaler for data normalization.
└── 📖 README.md           # Project documentation.

Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
Python 3.8 or newer installed.

pip (Python package installer).

2. Clone the Repository (Optional)
If you are using Git, you can clone the repository. Otherwise, ensure all files are in the same folder.

git clone <your-repo-url>
cd <your-repo-folder>

3. Install Dependencies
Install all the required Python libraries using pip:

pip install pandas scikit-learn streamlit joblib

4. Download the Dataset
Download the Kepler exoplanet dataset from Kaggle.

From the download, find the cumulative.csv file.

Place cumulative.csv in the root directory of the project.

How to Run the Application
The application is run in two stages.

Step 1: Train the Model
First, run the training script to process the dataset and create the model files (planet_model.joblib and scaler.joblib). Open your terminal in the project directory and run:

python train_model.py

You should see output confirming the model was trained successfully, with an accuracy report.

Step 2: Launch the Web App
Once the model is trained, launch the interactive web application. In the same terminal, run:

python -m streamlit run app.py

Your default web browser will automatically open a new tab with the application running.

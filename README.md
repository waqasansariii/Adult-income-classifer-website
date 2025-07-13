# 🧠 Adult Income Classifier — ML + Full-Stack App

This project predicts whether an individual's income exceeds $50K/year using demographic data. It includes:

- ✅ Machine Learning Model (XGBoost, trained on UCI Adult Dataset)
- 🌐 Full-Stack Web Application (React Frontend + Flask Backend)
- 📦 Pretrained model served via API

---

## 📁 Project Structure

adult-income-classifier/
├── ml-model/ # Model training, evaluation, and saving
│ ├── train_model.ipynb
│ ├── model_pipeline.pkl
│ └── ...
│
├── website/ # Full-stack web app
│ ├── backend/ # Flask API server
│ │ ├── app.py
│ │ └── saved_models/
│ │ └── best_xgboost_model_for_adult.pkl
│ │
│ └── frontend/ # React UI for input and prediction
│ ├── src/
│ └── public/ node modules 


---

## 🔍 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Goal**: Predict if `income >50K` or `<=50K` using features like age, education, occupation, etc.

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- Node.js + npm
- Git

---

### 🔁 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/adult-income-classifier.git
cd adult-income-classifier
📊 Model Info
Algorithm: XGBoostClassifier

Preprocessing: ColumnTransformer with OneHotEncoding + Feature Engineering

Evaluation: Accuracy, Precision, Recall, F1-Score

Serialized using joblib
✨ Features

Easy-to-use form to input user data

Real-time prediction using trained model

Fully responsive UI using Tailwind CSS

CORS-enabled Flask backend
📚 Tech Stack
Area	Tech
Frontend	React, Tailwind CSS
Backend	Flask, Python, pandas, XGBoost
Model	Scikit-learn, XGBoost, joblib
Tools	VSCode, Git, GitHub

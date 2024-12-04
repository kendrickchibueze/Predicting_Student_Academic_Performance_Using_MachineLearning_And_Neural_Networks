# Predicting Student Academic Performance Using Machine Learning and Neural Networks

## Project Description
This project focuses on predicting student academic performance using a combination of machine learning and deep learning techniques. By leveraging tools for data preprocessing, feature engineering, and model training, the project aims to identify patterns and insights that contribute to predicting academic outcomes effectively.

## Features
- **Data Preprocessing**: Handling missing data, normalization, and encoding categorical features.
- **Exploratory Data Analysis (EDA)**: Visualizing data distributions and identifying correlations.
- **Machine Learning Models**: Feature scaling and selection to prepare data for modeling.
- **Deep Learning Model**: Using TensorFlow/Keras to build and train a neural network for classification.
- **Evaluation Metrics**: Analyzing model performance using precision, recall, and other classification metrics.

---

## Tools and Libraries Used
The following tools and libraries were utilized:
- **Data Manipulation and Visualization**:
  - `pandas`
  - `missingno`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- **Machine Learning Preprocessing**:
  - `scikit-learn`:
    - `MinMaxScaler`, `StandardScaler`
    - `OneHotEncoder`, `LabelEncoder`
    - `train_test_split`, `ShuffleSplit`
- **Deep Learning**:
  - `TensorFlow` and `Keras`:
    - `Sequential`, `Dense`
    - `to_categorical`
- **Evaluation Metrics**:
  - `classification_report`
  - `precision_recall_curve`
  - `average_precision_score`

---

## Project Workflow
1. **Data Loading and Inspection**:
   - Load the dataset and inspect for missing values using `missingno`.
   - Analyze data structure, types, and distributions.

2. **Data Preprocessing**:
   - Handle missing values appropriately.
   - Encode categorical variables using `OneHotEncoder` or `LabelEncoder`.
   - Normalize features using `StandardScaler` or `MinMaxScaler`.

3. **Exploratory Data Analysis (EDA)**:
   - Use `matplotlib` and `seaborn` for visualizing correlations and distributions.
   - Identify relationships between features and target variables.

4. **Train-Test Split**:
   - Split the dataset into training and testing subsets using `train_test_split`.

5. **Model Development**:
   - **Machine Learning Models**:
     - Experiment with feature scaling and selection techniques.
   - **Deep Learning Models**:
     - Build a neural network using Keras with multiple layers and activation functions.
     - Train the model on the training data and evaluate on the testing data.

6. **Model Evaluation**:
   - Use metrics like precision, recall, and average precision score to evaluate model performance.
   - Generate classification reports and precision-recall curves.

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <[repository-url](https://github.com/kendrickchibueze/Predicting_Student_Academic_Performance_Using_MachineLearning_And_Neural_Networks.git)>
   cd <Predicting_Student_Academic_Performance_Using_MachineLearning_And_Neural_Networks>

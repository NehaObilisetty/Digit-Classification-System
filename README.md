# Digit-Classification-System
 Digit Classifier
Handwritten Digit Recognition with Machine Learning

# Introduction
This project is focused on recognizing handwritten digits using machine learning techniques. We'll be working with the popular digits dataset from scikit-learn, which contains images of hand-drawn digits along with their corresponding labels. The main goal is to develop and compare different machine learning models to find the most accurate one for digit recognition.

# Getting Started
Before running the code, ensure you have the necessary libraries installed. You can install the required libraries using pip:

# Dataset
We are using the scikit-learn library to load the digits dataset. This dataset includes the following information:

data: Features representing pixel values of images.
feature_names: Names of the features (pixel columns).
target: Labels representing the digit value (0-9).
target_names: Names of the target classes (digits 0-9).

# Data Exploration
We begin by exploring the dataset to gain a better understanding of its structure:

The dataset contains 1,797 samples and 65 columns.
There are 64 pixel columns representing the image's pixel values, and one column for the target labels.
No missing values are found in the dataset.

# Data Splitting
To train and evaluate our machine learning models, we split the dataset into training and testing sets:

Features (X) include all columns except the target.
Target (y) includes the digit labels.

# Model Building
We experiment with different machine learning models and evaluate their performance:

# Logistic Regression
We train a logistic regression model and achieve an accuracy of approximately 95.23%.
# Decision Tree
A decision tree classifier is trained, resulting in an accuracy of approximately 86.33%.
# Random Forest
A random forest classifier is used, which yields an accuracy of approximately 97.93%.
# Support Vector Machine (SVM)
An SVM classifier is trained, and it performs exceptionally well with an accuracy of approximately 98.73%.
# Naive Bayes
A Gaussian Naive Bayes classifier is trained, and it achieves an accuracy of approximately 82.35%.
# Model Tuning
We perform hyperparameter tuning to improve the model's performance:

# SVM Parameter Tuning
We perform a grid search to find the best combination of hyperparameters for the SVM model.
The optimized SVM model achieves an accuracy of approximately 99.05%.
# Random Forest Parameter Tuning
We also optimize the random forest model with hyperparameter tuning.
The tuned random forest model has an accuracy of approximately 97.14%.
# Conclusion
After extensive experimentation, we conclude that the Support Vector Machine (SVM) model performs the best for handwritten digit recognition, achieving an accuracy of approximately 99.05%. However, the Random Forest model also provides competitive accuracy of approximately 97.14%. Depending on your specific requirements and computational resources, you can choose the most suitable model for your application.

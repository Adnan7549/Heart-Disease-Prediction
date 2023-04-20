# Heart-Disease-Prediction

This is a machine learning system that predicts whether a person has heart disease or not based on several health-related features. The system uses three different algorithms: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Logistic Regression.

# Data Set
The data set used in this project is the Heart Disease Prediction dataset, which contains 303 instances and 14 attributes related to heart health. The data set was pre-processed using standard scaling and label encoding. Exploratory data analysis was performed to gain insights into the data, and data visualization was used to help understand the relationships between different features.

# Architecture
The architecture of the system was designed using the scikit-learn library in Python. The data was split into training and testing sets using a 75/25 ratio. The KNN algorithm was used with a value of k = 1 and Euclidean distance as the distance metric. The SVM algorithm was used with default parameters, and the Logistic Regression algorithm was used with default parameters as well.

# Evaluation
The performance of the system was evaluated using accuracy score, confusion matrix, and F1 score. The KNN algorithm had an accuracy score of 0.71, the SVM algorithm had an accuracy score of 0.79, and the Logistic Regression algorithm had an accuracy score of 0.84. The confusion matrix and F1 score were also used to evaluate the performance of each algorithm.

# Installation
To use this machine learning system, you will need to have Python 3 and the following Python libraries installed: pandas, numpy, matplotlib, seaborn, scikit-learn. To install these libraries, you can use pip:
**pip install pandas numpy matplotlib seaborn scikit-learn**

# Usage
To use this machine learning system, you can run the heart_disease_prediction.ipynb file using Jupyter Notebook. This will train the three algorithms and evaluate their performance on the test set. You can also modify the code to try different algorithms or hyperparameters.

# Credits
This project was inspired by the Heart Disease Prediction dataset on Kaggle. The code was developed by Mohammad Adnan.

# License
This project is licensed under the MIT License.

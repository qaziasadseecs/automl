# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
from sklearn.svm import OneClassSVM
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.models import save_model

# Preprocess data
def preprocess_data(data):
    # Convert non-numeric features to numeric representation
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column contains non-numeric data
            label_encoders[column] = LabelEncoder()  # Initialize LabelEncoder for the column
            data[column] = label_encoders[column].fit_transform(data[column])  # Convert non-numeric data to numeric
    
    # Ensure all remaining features are numeric
    # For example, handle missing values and ensure all columns have numeric data types
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
    
    # Handle missing values by filling with mean value
    data.fillna(data.mean(), inplace=True)
    
    # Return preprocessed data
    return data

def deep_learning_models(X_train, X_test, y_train, y_test):
    results = {}
    trained_models = {}
    # Assuming X_train and X_test have shapes (number_of_samples, number_of_features)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Simple 1D-CNN model
    print("Executing 1D-CNN:")
    cnn_model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=X_train_reshaped.shape[1:]),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
    results['1D-CNN'] = cnn_accuracy
    trained_models['1D-CNN'] = cnn_model
    
    # Another simpler sequential model
    print("Executing ANN:")
    simpler_model = Sequential([
        Dense(64, activation='relu', input_shape=X_train.shape[1:]),
        Dense(1, activation='sigmoid')
    ])
    simpler_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    simpler_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    simpler_accuracy = simpler_model.evaluate(X_test, y_test, verbose=0)[1]
    results['Simpler_Sequential'] = simpler_accuracy
    trained_models['Simpler_Sequential'] = simpler_model
    
    return results, trained_models


# Define AutoML function
def automl(X_train, X_test, y_train, y_test, filename):
    algorithms = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "kNN": KNeighborsClassifier(),
        "K-Means": KMeans(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "GBM": GradientBoostingClassifier(),  # Assuming GBM refers to Gradient Boosting
        "LightGBM": LGBMClassifier(),
    }

    results = {}
    model_filenames = {}  # To store filenames of top 3 models
    
    # Train and evaluate models
    for name, model in algorithms.items():
        print("Executing: {0}".format(name))
        if name in ["K-Means"]:
            # For clustering algorithms, fit on the entire dataset
            model.fit(X_train)
            # For clustering algorithms, use silhouette score
            score = silhouette_score(X_test, model.predict(X_test))
            results[name] = score
        else:
            # For supervised learning algorithms, fit on labeled data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
    # Add deep learning models to the results
    deep_learning_results, trained_models = deep_learning_models(X_train, X_test, y_train, y_test)
    results.update(deep_learning_results)
    print("\n" , results, "\n")
    # Sort results by score in descending order
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    return results

def access_model_selection():
    process_files(r"Data")

def process_files(folder_path):
    results = []
    performance = {}
    #folder_path = r"Data"
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            print("Working on:", filename)
            dataset = pd.read_csv(os.path.join(folder_path, filename))
            dataset = preprocess_data(dataset)
            
            print("Processing dataset:", filename)
            
            # Preprocess data
            preprocessed_data = preprocess_data(dataset)
            
            # Split data into features and target
            target_column = preprocessed_data.columns[-1]  # Assuming the last column is the target column
            X = preprocessed_data.drop(columns=[target_column])
            y = preprocessed_data[target_column]
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # AutoML
            print("Running AutoML...")
            scores = automl(X_train, X_test, y_train, y_test, filename)
            
            performance[filename] = scores
    
 

            # Convert results to DataFrame and store in performance.csv
            performance_df = pd.DataFrame.from_dict(scores, orient='index')
            performance_df.index.name = 'Filename'
            performance_df.to_csv("performance.csv")
    




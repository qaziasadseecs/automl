import os
import pandas as pd
import numpy as np
from pymfe.mfe import MFE
from scipy.stats import skew, kurtosis, f_oneway, entropy, moment, normaltest
from sklearn.decomposition import PCA
from sklearn.utils import check_array
import warnings


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.add(array, 0.0000001, casting="unsafe")
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def extract_features(data):
    features = {}
    features['mean'] = np.mean(data)
    features['normalized_mean'] = np.mean(data) / np.std(data)
    features['std'] = np.std(data)
    features['variance'] = np.var(data)
    features['kurtosis'] = kurtosis(data)
    features['skewness'] = skew(data)
    features['gini'] = gini(data)
    features['entropy'] = entropy(np.histogram(data)[0])
    features['normality_p_value'] = normaltest(data)[1]
    moments = moment(data, moment=[5, 6, 7, 8, 9, 10])
    features.update({f'moment_{i+5}': moment for i, moment in enumerate(moments)})
    return features


def extract_pymfe_meta_features(data):
    mfe = MFE(groups=["statistical", "info-theory"])
    mfe.fit(data)
    meta_features_list = mfe.extract(as_dict=True)
    # Extract dictionaries from the list
    meta_features = {}
    for i, feature_name in enumerate(meta_features_list[0]):
        meta_features[feature_name] = meta_features_list[1][i]
    
    # Handle the case where some meta-features cannot be calculated
    for key, value in meta_features.items():
        if isinstance(value, dict):
            meta_features[key] = {k: np.nan for k in value.keys()}
    
    return meta_features
from sklearn.preprocessing import LabelEncoder
import pandas as pd
def gen_meta_features(X, filename):
    X = check_array(X)
    meta_vec = []
    meta_vec_names = []
    n_samples, n_features = X.shape[0], X.shape[1]
    
    meta_vec.append(n_samples)
    meta_vec.append(n_features)
    meta_vec_names.append('n_samples')
    meta_vec_names.append('n_features')
    
    # Extract statistical features
    all_mean = []
    all_normalized_mean = []
    all_std = []
    all_variance = []
    all_kurtosis = []
    all_skewness = []
    all_gini = []
    all_entropy = []
    all_normality_p_value = []
    all_moments = [[] for _ in range(6)]  # List to store moments for each order
    
    for i in range(n_features):
        features = extract_features(X[:, i])
        all_mean.append(features['mean'])
        all_normalized_mean.append(features['normalized_mean'])
        all_std.append(features['std'])
        all_variance.append(features['variance'])
        all_kurtosis.append(features['kurtosis'])
        all_skewness.append(features['skewness'])
        all_gini.append(features['gini'])
        all_entropy.append(features['entropy'])
        all_normality_p_value.append(features['normality_p_value'])
        for j, moment_order in enumerate(range(5, 11)):
            all_moments[j].append(features[f'moment_{moment_order}'])
    
    # Aggregate statistical features
    meta_vec.extend([
        np.mean(all_mean),
        np.mean(all_normalized_mean),
        np.mean(all_std),
        np.mean(all_variance),
        np.mean(all_kurtosis),
        np.mean(all_skewness),
        np.mean(all_gini),
        np.mean(all_entropy),
        np.mean(all_normality_p_value)
    ])
    for moments in all_moments:
        meta_vec.append(np.mean(moments))
    
    # PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    pca_features = pca.explained_variance_ratio_
    meta_vec.extend(pca_features)
    meta_vec_names.extend([f'pca_expl_ratio_{i+1}' for i in range(len(pca_features))])
    
    # Extract meta-features using pymfe
    pymfe_meta_features = extract_pymfe_meta_features(X)
    pymfe_meta_feature_names = list(pymfe_meta_features.keys())
    for key in pymfe_meta_feature_names:
        value = pymfe_meta_features[key]
        if isinstance(value, dict):
            meta_vec.append(np.mean(list(value.values())))
        else:
            meta_vec.append(value)
    meta_vec_names.extend(pymfe_meta_feature_names)

    
    return meta_vec, meta_vec_names

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

def process_datasets(folder_path, output_csv):
    # Suppress warnings
    warnings.filterwarnings("ignore")
    # Check if the output CSV file already exists
    if os.path.exists(output_csv):
        existing_data = pd.read_csv(output_csv)
        existing_files = set(existing_data['Filename'])
    else:
        existing_files = set()

    all_meta_features = []
    all_meta_feature_names = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            if filename in existing_files:
                print(f"Skipping {filename}, entry already exists in {output_csv}")
                continue
            print("Working on: ", filename)
            dataset = pd.read_csv(os.path.join(folder_path, filename))
            dataset = preprocess_data(dataset)
            meta_features, meta_feature_names = gen_meta_features(dataset.values, filename)
            all_meta_features.append(meta_features)
            all_meta_feature_names.append(meta_feature_names)
    
    # Construct the DataFrame
    meta_features_df = pd.DataFrame(columns=['Filename'] + all_meta_feature_names[0])
    for features, names in zip(all_meta_features, all_meta_feature_names):
        features_dict = {'Filename': filename}
        features_dict.update({name: value for name, value in zip(names, features)})
        meta_features_df = meta_features_df.append(features_dict, ignore_index=True)
    
   # Save DataFrame to CSV
    if os.path.exists(output_csv):
        meta_features_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        meta_features_df.to_csv(output_csv, index=False)




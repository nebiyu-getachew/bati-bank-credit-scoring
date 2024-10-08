import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import scorecardpy as sc


def create_aggregate_features(data):
    grouped = data.groupby('CustomerId')
    data['TotalTransactionAmount'] = grouped['Amount'].transform('sum')
    data['AverageTransactionAmount'] = grouped['Amount'].transform('mean')
    data['TransactionCount'] = grouped['TransactionId'].transform('count')
    data['StdDevTransactionAmount'] = grouped['Amount'].transform('std')
    return data

def extract_time_features(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year
    return data


def encode_categorical_features(data, categorical_columns):
    """Encode categorical features using one-hot encoding."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_data = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
    return pd.concat([data.drop(columns=categorical_columns), encoded_data], axis=1)

def handle_missing_values(data):
    num_imputer = SimpleImputer(strategy='median')
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])
    
    return data

def normalize_features(data):
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['int64','int32', 'float64']).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

def rfms_score(data):
    """
    Calculate RFMS score for each customer.
    RFMS: Recency, Frequency, Monetary, Size
    """
    # Assuming TransactionStartTime is already in datetime format
    current_date = data['TransactionStartTime'].max()
    
    customer_metrics = data.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (current_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': ['sum', 'mean'],  # Monetary and Size
    })
    
    customer_metrics.columns = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg']
    
    # Normalize the metrics
    for col in customer_metrics.columns:
        customer_metrics[f'{col}_Normalized'] = (customer_metrics[col] - customer_metrics[col].min()) / (customer_metrics[col].max() - customer_metrics[col].min())
    
    # Calculate RFMS score (you may adjust the weights as needed)
    customer_metrics['RFMS_Score'] = (
        0.25 * (1 - customer_metrics['Recency_Normalized']) +  # Inverse of Recency
        0.25 * customer_metrics['Frequency_Normalized'] +
        0.25 * customer_metrics['MonetaryTotal_Normalized'] +
        0.25 * customer_metrics['MonetaryAvg_Normalized']
    )
    
    return customer_metrics

def plot_rfms_distributions(customer_metrics):
    plt.figure(figsize=(15, 12))
    
    metrics = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryAvg', 'RFMS_Score']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        sns.histplot(customer_metrics[metric], kde=True, bins=30, color='blue')
        plt.title(f'{metric} Distribution')
        plt.xlabel(metric)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()


def assign_good_bad_label(data, rfms_scores, threshold=0.4):
    """
    Assign good/bad labels based on RFMS score per customer
    """
    # Merge RFMS scores with the original dataframe
    customer_labels = rfms_scores['RFMS_Score'].reset_index()
    customer_labels['label'] = np.where(customer_labels['RFMS_Score'] > threshold, 'good', 'bad')
    
    # Merge labels back to the original dataframe
    data = data.merge(customer_labels[['CustomerId', 'RFMS_Score', 'label']], on='CustomerId', how='left')
    
    return data


def woe_binning(data, target_col, features):
    """
    Perform Weight of Evidence (WoE) binning on specified features.
    
    :param data: DataFrame containing the features and target variable
    :param target_col: Name of the target column
    :param features: List of feature names to perform WoE binning on
    :return: DataFrame with WoE binned features
    """
    bins = sc.woebin(data, y=target_col, x=features)
    woe_data = sc.woebin_ply(data, bins)

    # Extract and print IV for each feature
    iv_values = {}
    for feature in features:
        iv_values[feature] = bins[feature]['total_iv'].values[0]
        print(f"IV for {feature}: {iv_values[feature]}")


    return woe_data, iv_values


def plot_woe_binning(bins, feature):
    """
    Plot WoE binning results for a specific feature.
    """
    plt.figure(figsize=(10, 6))
    sc.woebin_plot(bins[feature])
    plt.title(f'WoE Binning Plot for {feature}')
    plt.tight_layout()
    plt.show()
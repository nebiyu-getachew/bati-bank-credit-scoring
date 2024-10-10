# Bati Bank Credit Scoring

## Project Overview
This project involves building a **Credit Scoring Model** for Bati Bank in collaboration with an eCommerce platform. The model will allow Bati Bank to assess the creditworthiness of customers who wish to use the **Buy Now Pay Later** (BNPL) service offered by the eCommerce company.

The key goal of the project is to develop a data-driven solution that accurately assigns a credit score and risk probability for customers, helping the bank mitigate potential risks associated with offering credit to its customers.

### Project Objectives:
1. Define a proxy variable to categorize users as **high risk** (bad) or **low risk** (good).
2. Select features that predict the likelihood of default.
3. Develop a **Credit Scoring Model** that assigns a risk probability to new customers.
4. Predict an optimal loan amount and duration for each customer.

## Data and Features
The project uses data provided by the eCommerce platform, with key fields including:
- **Transaction ID**: Unique identifier for each transaction.
- **Customer ID**: Unique customer identifier.
- **Amount**: Value of each transaction.
- **Product ID** and **Category**: The item purchased and its category.
- **Transaction Start Time**: The timestamp when the transaction was initiated.
- **Fraud Result**: Whether the transaction was marked as fraudulent.

A detailed description of all the data fields can be found in the project documentation.

## Key Steps

### 1. Exploratory Data Analysis (EDA)
- Understanding the dataset, including summary statistics, distribution of features, and correlation analysis.
- Identifying missing values and outliers.

### 2. Feature Engineering
- Creating aggregate features such as total transaction amount, average transaction amount, transaction count, and variability of transaction amounts per customer.
- Encoding categorical variables and handling missing values.
- Normalizing and standardizing numerical features.

### 3. Model Development
- **Model Selection**: Logistic Regression, Random Forest, and Gradient Boosting Machines.
- **Model Training**: Splitting the data into training and testing sets.
- **Model Tuning**: Using hyperparameter optimization techniques such as Grid Search and Random Search.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

### 4. Model Deployment
- The trained model will be deployed via a REST API to enable real-time credit scoring predictions for the bank’s customers.

## Project Deliverables
- **Model Files**: Trained model files available in the `models/` directory.
- **Scripts**: The Python scripts used for data preprocessing, model training, and evaluation.
- **API Code**: The Flask/FastAPI code for serving the model via REST API.
- **Reports**: Detailed report of the EDA, model development, and evaluation in the `reports/` folder.

## Getting Started

### Prerequisites
To run this project locally, you will need the following installed on your machine:
- Python 3.8+
- Libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `Flask` or `FastAPI`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nebiyu-ethio/bati-bank-credit-scoring.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_prepare_data():
    """Load and prepare the training data with the top 5 features."""
    
    # Load the training data
    train_df = pd.read_csv('train.csv')
    
    # Select the top 5 features based on EDA analysis
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
    target = 'SalePrice'
    
    # Handle missing values for these specific features
    # GarageCars and GarageArea: fill with 0 (no garage)
    train_df['GarageCars'] = train_df['GarageCars'].fillna(0)
    train_df['GarageArea'] = train_df['GarageArea'].fillna(0)
    
    # TotalBsmtSF: fill with 0 (no basement)
    train_df['TotalBsmtSF'] = train_df['TotalBsmtSF'].fillna(0)
    
    # OverallQual and GrLivArea should not have missing values, but check anyway
    train_df['OverallQual'] = train_df['OverallQual'].fillna(train_df['OverallQual'].median())
    train_df['GrLivArea'] = train_df['GrLivArea'].fillna(train_df['GrLivArea'].median())
    
    # Extract features and target
    X = train_df[features]
    y = train_df[target]
    
    # Apply log transformation to target (as suggested in EDA)
    y_log = np.log1p(y)
    
    return X, y_log, features

def train_model():
    """Train a linear regression model and save it."""
    
    print("Loading and preparing data...")
    X, y, features = load_and_prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training the model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Convert back from log scale
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    y_pred_train_actual = np.expm1(y_pred_train)
    y_pred_test_actual = np.expm1(y_pred_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Training RMSE: ${train_rmse:,.2f}")
    print(f"Testing RMSE: ${test_rmse:,.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Save the model and scaler
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/house_price_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    
    # Save feature names
    with open('model/features.txt', 'w') as f:
        f.write('\n'.join(features))
    
    print(f"\nModel saved successfully!")
    print(f"Features used: {', '.join(features)}")
    
    return model, scaler, features

if __name__ == "__main__":
    model, scaler, features = train_model()


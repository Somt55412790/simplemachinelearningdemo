"""
Model service module for house price prediction.
This module handles all machine learning operations and model management.
"""

import joblib
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HousePriceModel:
    """House price prediction model service."""
    
    def __init__(self, model_dir: str = 'model'):
        """Initialize the model service.
        
        Args:
            model_dir: Directory containing the trained model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.features = None
        self.is_loaded = False
        
        # Feature descriptions for API documentation
        self.feature_descriptions = {
            'OverallQual': {
                'description': 'Overall material and finish quality',
                'type': 'integer',
                'min': 1,
                'max': 10,
                'example': 7
            },
            'GrLivArea': {
                'description': 'Above ground living area in square feet',
                'type': 'integer',
                'min': 0,
                'max': 10000,
                'example': 1500
            },
            'GarageCars': {
                'description': 'Size of garage in car capacity',
                'type': 'integer',
                'min': 0,
                'max': 4,
                'example': 2
            },
            'GarageArea': {
                'description': 'Size of garage in square feet',
                'type': 'integer',
                'min': 0,
                'max': 2000,
                'example': 500
            },
            'TotalBsmtSF': {
                'description': 'Total square feet of basement area',
                'type': 'integer',
                'min': 0,
                'max': 5000,
                'example': 800
            }
        }
        
        # Load the model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model, scaler, and feature names.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'house_price_model.pkl')
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler file not found: {scaler_path}")
                return False
            
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
            
            # Load features
            features_path = os.path.join(self.model_dir, 'features.txt')
            if not os.path.exists(features_path):
                logger.error(f"Features file not found: {features_path}")
                return False
            
            with open(features_path, 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            
            logger.info(f"Features loaded: {self.features}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def validate_input(self, data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate input data for prediction.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.is_loaded:
            return False, "Model not loaded"
        
        # Check if all required features are present
        for feature in self.features:
            if feature not in data:
                return False, f"Missing required feature: {feature}"
        
        # Validate each feature
        for feature, value in data.items():
            if feature in self.feature_descriptions:
                desc = self.feature_descriptions[feature]
                
                # Check type
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return False, f"Invalid value for {feature}: must be a number"
                
                # Check range
                if value < desc['min'] or value > desc['max']:
                    return False, f"Value for {feature} must be between {desc['min']} and {desc['max']}"
        
        return True, None
    
    def predict(self, data: Dict) -> Dict:
        """Make a house price prediction.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Validate input
        is_valid, error_msg = self.validate_input(data)
        if not is_valid:
            raise ValueError(error_msg)
        
        try:
            # Extract feature values in the correct order
            feature_values = []
            for feature in self.features:
                value = float(data[feature])
                feature_values.append(value)
            
            # Convert to numpy array and reshape
            input_data = np.array(feature_values).reshape(1, -1)
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction (model predicts log price)
            log_prediction = self.model.predict(input_scaled)[0]
            
            # Convert back from log scale
            prediction = np.expm1(log_prediction)
            
            # Calculate confidence interval (simple approach)
            # In a more sophisticated model, you'd use prediction intervals
            confidence_margin = prediction * 0.15  # ±15% as rough estimate
            
            return {
                'predicted_price': float(prediction),
                'formatted_price': f"${prediction:,.2f}",
                'confidence_interval': {
                    'lower': float(prediction - confidence_margin),
                    'upper': float(prediction + confidence_margin),
                    'formatted_lower': f"${(prediction - confidence_margin):,.2f}",
                    'formatted_upper': f"${(prediction + confidence_margin):,.2f}"
                },
                'features_used': dict(zip(self.features, feature_values)),
                'model_version': "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_loaded': self.is_loaded,
            'features': self.features,
            'feature_descriptions': self.feature_descriptions,
            'model_type': 'Linear Regression',
            'version': '1.0.0',
            'training_data': 'Ames Housing Dataset'
        }
    
    def health_check(self) -> Dict:
        """Perform a health check on the model service.
        
        Returns:
            Dictionary containing health status
        """
        status = "healthy" if self.is_loaded else "unhealthy"
        
        # Test prediction with sample data
        can_predict = False
        if self.is_loaded:
            try:
                sample_data = {
                    'OverallQual': 7,
                    'GrLivArea': 1500,
                    'GarageCars': 2,
                    'GarageArea': 500,
                    'TotalBsmtSF': 800
                }
                self.predict(sample_data)
                can_predict = True
            except Exception as e:
                logger.error(f"Health check prediction failed: {str(e)}")
        
        return {
            'status': status,
            'model_loaded': self.is_loaded,
            'can_predict': can_predict,
            'features_count': len(self.features) if self.features else 0,
            'timestamp': pd.Timestamp.now().isoformat()
        }

# Global model instance
model_service = HousePriceModel()

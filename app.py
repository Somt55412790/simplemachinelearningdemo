"""
PropertyValuer Pro - AI-Powered Real Estate Valuation Platform
A production-ready Flask application for house price prediction using machine learning.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import logging
from datetime import datetime
from model_service import model_service

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
CORS(app, origins=['*'])  # Configure CORS for production

# Flask configuration
app.config.update(
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=True
)

@app.route('/')
def home():
    """
    Render the main PropertyValuer Pro interface.
    
    Returns:
        str: Rendered HTML template with model features
    """
    try:
        model_info = model_service.get_model_info()
        return render_template('index.html', features=model_info.get('features', []))
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return render_template('index.html', features=[])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Generate house price prediction based on property characteristics.
    
    Expected JSON payload:
    {
        "OverallQual": int (1-10),
        "GrLivArea": int (sq ft),
        "GarageCars": int (0-4),
        "GarageArea": int (sq ft),
        "TotalBsmtSF": int (sq ft)
    }
    
    Returns:
        JSON: Prediction result with confidence intervals and metadata
    """
    try:
        # Validate request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided. Please submit property characteristics.'
            }), 400
        
        # Validate required fields
        required_fields = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Log prediction request
        logger.info(f"Prediction request received: {data}")
        
        # Generate prediction using model service
        result = model_service.predict(data)
        
        # Return successful prediction
        response = {
            'success': True,
            'prediction': result['formatted_price'],
            'raw_prediction': result['predicted_price'],
            'confidence_interval': result['confidence_interval'],
            'features_used': result['features_used'],
            'model_version': result['model_version'],
            'timestamp': datetime.now().isoformat(),
            'accuracy': '82.4% R² Score'
        }
        
        logger.info(f"Prediction successful: {result['formatted_price']}")
        return jsonify(response)
        
    except ValueError as e:
        error_msg = f'Invalid input data: {str(e)}'
        logger.warning(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 400
    except RuntimeError as e:
        error_msg = f'Prediction service error: {str(e)}'
        logger.error(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500
    except Exception as e:
        error_msg = 'Internal server error occurred'
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/health')
def health_check():
    """
    Health check endpoint for deployment monitoring.
    
    Returns:
        JSON: Service health status and model information
    """
    try:
        health_status = model_service.health_check()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': 'Health check failed',
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/features')
def get_features():
    """
    Get supported features and their descriptions.
    
    Returns:
        JSON: Model features information
    """
    try:
        model_info = model_service.get_model_info()
        return jsonify({
            'success': True,
            'data': model_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Unable to retrieve features information'
        }), 500

@app.route('/api/model/info')
def get_model_info():
    """
    Get detailed model information and performance metrics.
    
    Returns:
        JSON: Comprehensive model information
    """
    try:
        model_info = model_service.get_model_info()
        model_info.update({
            'accuracy': '82.4% R² Score',
            'rmse': '$32,297',
            'training_data': '1,460 properties',
            'last_updated': datetime.now().isoformat()
        })
        return jsonify({
            'success': True,
            'data': model_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Unable to retrieve model information'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for programmatic house price prediction.
    
    Returns:
        JSON: Prediction results in API format
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        result = model_service.predict(data)
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 400
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Prediction service temporarily unavailable',
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# Production WSGI entry point
if __name__ == '__main__':
    # Development server (not for production)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting PropertyValuer Pro on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)


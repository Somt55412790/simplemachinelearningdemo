# 🏠 House Price Predictor - ML Web Application

A machine learning web application that predicts house prices using 5 key characteristics, built with Flask and deployed on Railway using Docker.

## 🎯 Overview

This application uses a Linear Regression model trained on the Ames Housing dataset to predict house prices. The model was developed based on comprehensive EDA analysis and focuses on the 5 most predictive features.

## ✨ Features

- **Interactive Web Interface**: Beautiful, responsive UI for house price predictions
- **RESTful API**: Complete API for integration with other applications
- **Batch Predictions**: Support for multiple house predictions in a single request
- **Confidence Intervals**: Provides prediction ranges with confidence estimates
- **Docker Support**: Containerized for easy deployment
- **Railway Ready**: Pre-configured for Railway deployment
- **Health Monitoring**: Built-in health checks and monitoring endpoints

## 🚀 Live Demo

[Visit the Live Application](https://your-app-name.railway.app)

## 📊 Model Details

### Features Used
The model uses these 5 key features identified through EDA analysis:

1. **Overall Quality** (1-10): Overall material and finish quality
2. **Living Area** (sq ft): Above ground living area
3. **Garage Cars** (0-4): Garage capacity in number of cars
4. **Garage Area** (sq ft): Size of garage in square feet
5. **Basement Area** (sq ft): Total basement square footage

### Performance Metrics
- **RMSE**: ~$32,000
- **R² Score**: ~0.85
- **Confidence Interval**: ±15%
- **Training Data**: 1,460 houses from Ames, Iowa

## 🛠️ Technology Stack

- **Backend**: Python, Flask, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **ML Framework**: scikit-learn, pandas, numpy
- **Containerization**: Docker
- **Deployment**: Railway
- **API**: RESTful with JSON responses

## 🏃‍♂️ Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd house-price-predictor

# Build and run with Docker
docker build -t house-price-predictor .
docker run -p 5000:5000 house-price-predictor
```

### Option 2: Local Development

```bash
# Clone the repository
git clone <repository-url>
cd house-price-predictor

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the application
python app.py
```

Visit `http://localhost:5000` to use the application.

## 🔌 API Usage

### Single Prediction
```python
import requests

data = {
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "GarageArea": 500,
    "TotalBsmtSF": 800
}

response = requests.post("http://localhost:5000/api/predict", json=data)
result = response.json()
print(f"Predicted Price: {result['data']['formatted_price']}")
```

### Batch Predictions
```python
batch_data = {
    "predictions": [
        {
            "OverallQual": 7,
            "GrLivArea": 1500,
            "GarageCars": 2,
            "GarageArea": 500,
            "TotalBsmtSF": 800
        },
        {
            "OverallQual": 8,
            "GrLivArea": 2000,
            "GarageCars": 3,
            "GarageArea": 750,
            "TotalBsmtSF": 1000
        }
    ]
}

response = requests.post("http://localhost:5000/api/batch-predict", json=batch_data)
```

## 📚 API Documentation

Complete API documentation is available in [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Available Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /api/model/info` - Model information
- `POST /api/predict` - Single prediction
- `POST /api/batch-predict` - Batch predictions

## 🚀 Deployment on Railway

### Step 1: Prepare Your Repository
Ensure your code is in a Git repository with all the files from this project.

### Step 2: Deploy to Railway
1. Visit [Railway](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect the Dockerfile and deploy

### Step 3: Configure Environment (Optional)
No additional environment variables needed for basic deployment.

### Step 4: Access Your App
Railway will provide a URL like `https://your-app-name.railway.app`

## 📁 Project Structure

```
house-price-predictor/
├── app.py                 # Main Flask application
├── model_service.py       # ML model service layer
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── railway.json          # Railway deployment config
├── .dockerignore         # Docker ignore rules
├── templates/
│   └── index.html        # Web interface
├── model/                # Trained model files (created after training)
│   ├── house_price_model.pkl
│   ├── scaler.pkl
│   └── features.txt
├── train.csv             # Training data
├── README.md             # This file
└── API_DOCUMENTATION.md  # Complete API docs
```

## 🧪 Testing the Application

### Test the Web Interface
1. Open `http://localhost:5000`
2. Enter sample values:
   - Overall Quality: 7
   - Living Area: 1500
   - Garage Cars: 2
   - Garage Area: 500
   - Basement Area: 800
3. Click "Predict House Price"

### Test the API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 7,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "GarageArea": 500,
    "TotalBsmtSF": 800
  }'
```

## 🔧 Development

### Adding New Features
1. Update `model_service.py` for ML-related changes
2. Update `app.py` for new API endpoints
3. Update `templates/index.html` for UI changes
4. Update `requirements.txt` for new dependencies

### Model Retraining
```bash
python train_model.py
```
This will retrain the model and save updated model files.

## 📈 Performance Monitoring

The application includes several monitoring endpoints:

- `/health` - Basic health check
- `/api/model/info` - Model information and status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ames Housing Dataset from Kaggle
- Flask and scikit-learn communities
- Railway for easy deployment platform

## 📧 Support

For questions or issues:
1. Check the [API Documentation](API_DOCUMENTATION.md)
2. Open an issue in the repository
3. Contact Somtochukwuazubike@outlook.com.au

---

**Built with ❤️ using Python, Flask, and Machine Learning**

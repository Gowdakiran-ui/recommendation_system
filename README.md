# Neural Recommendation System

A PyTorch-based neural collaborative filtering recommendation system with MLflow tracking and CI/CD pipeline.

## 🎯 Project Overview

This project implements a neural network-based recommender system that:
- Trains on MovieLens dataset with 5M+ ratings
- Uses user and movie embeddings with neural networks
- Tracks experiments with MLflow
- Provides comprehensive inference capabilities
- Includes full CI/CD pipeline with Jenkins

## 📁 Project Structure

```
recommendation_system/
├── train.py              # Main training script
├── inference.py          # Inference class for predictions
├── simple_inference.py   # Demo inference script
├── quick_predict.py      # Quick prediction utilities
├── testing.py           # Comprehensive test suite
├── movies_ratings_cleaned.csv  # Training dataset
├── mlruns/              # MLflow experiment tracking
├── Jenkinsfile          # CI/CD pipeline configuration
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   python train.py
   ```

3. **Run Inference**
   ```bash
   python simple_inference.py
   ```

4. **Run Tests**
   ```bash
   python testing.py
   ```

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -t neural-recommender .
   ```

2. **Run Container**
   ```bash
   docker run -p 5000:5000 neural-recommender
   ```

## 🤖 Model Architecture

- **Embedding Layers**: User and movie embeddings (50 dimensions)
- **Neural Network**: 2 hidden layers (128 → 64 → 1)
- **Training**: 25 epochs, batch size 512, learning rate 0.001
- **Performance**: RMSE ~1.06 on test set

## 📊 Features

### Training (`train.py`)
- ✅ Random sampling of 150K rows for performance
- ✅ LabelEncoder preprocessing for user/movie IDs
- ✅ PyTorch Dataset and DataLoader implementation
- ✅ Neural collaborative filtering model
- ✅ MLflow experiment tracking
- ✅ Model registration and versioning

### Inference (`inference.py`)
- ✅ Model loading from MLflow registry or runs
- ✅ Single user-item predictions
- ✅ Batch predictions
- ✅ Top-N recommendations
- ✅ Comprehensive error handling

### Testing (`testing.py`)
- ✅ Single prediction tests
- ✅ Batch prediction tests
- ✅ Recommendation generation tests
- ✅ Edge case handling
- ✅ Performance analytics

## 🔧 CI/CD Pipeline

The Jenkins pipeline includes:

### Stages
1. **🔍 Project Scan & Setup** - Environment preparation
2. **🐍 Python Environment Setup** - Dependency installation
3. **📋 Code Quality Analysis** - Formatting, linting, type checking
4. **📊 Data Validation** - Dataset integrity checks
5. **🧪 Unit Tests** - Comprehensive test suite execution
6. **🤖 Model Training & Validation** - ML model pipeline
7. **🔬 Model Validation** - Inference and performance testing
8. **📦 Build Artifacts** - Docker images and model packaging
9. **🚀 Deployment** - Environment-specific deployment
10. **📊 Performance Monitoring** - Metrics and reporting

### Parameters
- `ENVIRONMENT`: Target deployment environment
- `RETRAIN_MODEL`: Force model retraining
- `RUN_FULL_TESTS`: Execute complete test suite
- `SAMPLE_SIZE`: Training data sample size

### Quality Gates
- Code formatting with Black
- Linting with flake8
- Type checking with MyPy
- Test coverage minimum 80%
- Model validation requirements

## 📈 MLflow Integration

### Experiment Tracking
- Hyperparameters logging
- Training loss per epoch
- Final test RMSE
- Model artifacts
- Environment information

### Model Registry
- Model versioning
- Stage management (staging/production)
- Model lineage tracking
- Performance comparison

## 🧪 Testing Framework

### Test Categories
- **Single Predictions**: Individual user-item rating predictions
- **Batch Predictions**: Multiple predictions processing
- **Recommendations**: Top-N movie recommendations
- **Edge Cases**: Error handling and boundary conditions

### Test Results
- Overall Success Rate: 78.6%
- Average Rating: 4.071
- Rating Range: 2.561 - 5.045

## 🔒 Security Features

- Non-root Docker user
- Input validation
- Error handling
- Secure secret management
- Container health checks

## 🌍 Environment Configuration

### Development
```bash
export MLFLOW_TRACKING_URI=file:./mlruns
export ENVIRONMENT=development
```

### Staging
```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export ENVIRONMENT=staging
```

### Production
```bash
export MLFLOW_TRACKING_URI=https://mlflow.company.com
export ENVIRONMENT=production
```

## 📋 API Usage

### Basic Inference
```python
from inference import ModelInference

# Initialize inference
inference = ModelInference(model_name="neural_recommender")
inference.load_model_from_registry("latest")
inference.prepare_encoders()

# Single prediction
rating = inference.predict_rating(user_id=123, movie_id=456)

# Recommendations
recommendations = inference.recommend_movies_for_user(user_id=123, top_k=10)
```

### Batch Processing
```python
# Batch predictions
user_movie_pairs = [(1, 10), (2, 20), (3, 30)]
predictions = inference.predict_batch(user_movie_pairs)
```

## 🔧 Troubleshooting

### Common Issues
1. **Model Loading Failed**: Check MLflow tracking URI and model registry
2. **Encoder Errors**: Ensure data preprocessing matches training
3. **Memory Issues**: Reduce batch size or sample size
4. **Docker Build Failed**: Check system dependencies

### Debug Commands
```bash
# Check MLflow runs
mlflow ui --backend-store-uri file:./mlruns

# Validate model
python -c "from inference import ModelInference; ModelInference().load_model_from_registry()"

# Check data integrity
python -c "import pandas as pd; print(pd.read_csv('movies_ratings_cleaned.csv').info())"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run tests: `python testing.py`
4. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Kiran Gowda - Initial development
- AI Assistant - Pipeline automation

## 🔗 Links

- [GitHub Repository](https://github.com/Gowdakiran-ui/recommendation_system.git)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
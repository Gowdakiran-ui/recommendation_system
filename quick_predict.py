"""
Quick Model Loader and Predictor
This script can load any trained model by run ID and make predictions
"""

import sys
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def quick_predict(run_id=None, user_id=None, movie_id=None, top_k=5):
    """
    Quick function to load model and make predictions
    
    Args:
        run_id: MLflow run ID (if None, will use latest from registry)
        user_id: User ID for recommendation (optional)
        movie_id: Movie ID for single prediction (optional) 
        top_k: Number of top recommendations to show
    """
    
    print(f"🚀 Loading Neural Recommender Model")
    print(f"Run ID: {run_id}")
    print("=" * 60)
    
    # Load model
    try:
        if run_id:
            model_uri = f"runs:/{run_id}/model"
            print(f"Loading from specific run: {run_id}")
        else:
            model_uri = "models:/neural_recommender/latest"
            print("Loading latest model from registry")
            
        model = mlflow.pytorch.load_model(model_uri)
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Prepare encoders (same as training)
    print("\n📊 Preparing data encoders...")
    df = pd.read_csv("movies_ratings_cleaned.csv")
    df = df.sample(n=150000, random_state=42)
    
    # Handle column mapping
    expected_columns = ['userId', 'movieId', 'rating']
    if not all(col in df.columns for col in expected_columns):
        column_mapping = {}
        for col in df.columns:
            if 'user' in col.lower():
                column_mapping[col] = 'userId'
            elif 'movie' in col.lower() or 'item' in col.lower():
                column_mapping[col] = 'movieId'
            elif 'rating' in col.lower():
                column_mapping[col] = 'rating'
        if column_mapping:
            df = df.rename(columns=column_mapping)
    
    df = df[['userId', 'movieId', 'rating']].copy().dropna()
    
    # Create encoders
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    user_encoder.fit(df['userId'])
    movie_encoder.fit(df['movieId'])
    
    print(f"✅ Ready! {len(user_encoder.classes_)} users, {len(movie_encoder.classes_)} movies")
    
    # Single prediction if both user_id and movie_id provided
    if user_id is not None and movie_id is not None:
        print(f"\n🎯 Single Prediction")
        print(f"User {user_id} → Movie {movie_id}")
        
        try:
            user_encoded = user_encoder.transform([user_id])[0]
            movie_encoded = movie_encoder.transform([movie_id])[0]
            
            user_tensor = torch.LongTensor([user_encoded])
            movie_tensor = torch.LongTensor([movie_encoded])
            
            model.eval()
            with torch.no_grad():
                prediction = model(user_tensor, movie_tensor)
                rating = prediction.item()
            
            # Check if actual rating exists
            actual = df[(df['userId'] == user_id) & (df['movieId'] == movie_id)]['rating']
            actual_str = f" (Actual: {actual.iloc[0]:.1f})" if len(actual) > 0 else ""
            
            print(f"🌟 Predicted Rating: {rating:.2f}{actual_str}")
            
        except ValueError:
            print("❌ User or Movie not found in training data")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Recommendations if user_id provided
    if user_id is not None:
        print(f"\n🎬 Top {top_k} Recommendations for User {user_id}")
        print("-" * 50)
        
        try:
            user_encoded = user_encoder.transform([user_id])[0]
            all_movies = movie_encoder.classes_
            
            # Batch prediction for all movies
            user_tensor = torch.LongTensor([user_encoded] * len(all_movies))
            movie_tensors = torch.LongTensor(range(len(all_movies)))
            
            model.eval()
            with torch.no_grad():
                predictions = model(user_tensor, movie_tensors)
            
            # Create and sort recommendations
            recommendations = [(all_movies[i], predictions[i].item()) 
                             for i in range(len(all_movies))]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for i, (movie_id, rating) in enumerate(recommendations[:top_k], 1):
                print(f"  {i:2d}. Movie {movie_id}: {rating:.2f} ⭐")
                
        except ValueError:
            print(f"❌ User {user_id} not found in training data")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show some random predictions if no specific user/movie provided
    if user_id is None and movie_id is None:
        print(f"\n🎲 Random Sample Predictions")
        print("-" * 40)
        
        sample_users = df['userId'].unique()[:3]
        sample_movies = df['movieId'].unique()[:3]
        
        model.eval()
        for user in sample_users:
            print(f"\nUser {user}:")
            for movie in sample_movies:
                try:
                    user_encoded = user_encoder.transform([user])[0]
                    movie_encoded = movie_encoder.transform([movie])[0]
                    
                    user_tensor = torch.LongTensor([user_encoded])
                    movie_tensor = torch.LongTensor([movie_encoded])
                    
                    with torch.no_grad():
                        prediction = model(user_tensor, movie_tensor)
                        rating = prediction.item()
                    
                    print(f"  Movie {movie}: {rating:.2f}")
                    
                except Exception as e:
                    print(f"  Movie {movie}: Error")
    
    print(f"\n✨ Prediction completed!")

if __name__ == "__main__":
    # Example usage - you can modify these parameters
    
    # Use the latest run ID from your training
    latest_run_id = "ef7c3a3244654413b979af93d39253e9"
    
    # Example 1: Single prediction
    print("Example 1: Single User-Movie Prediction")
    quick_predict(run_id=latest_run_id, user_id=645, movie_id=380)
    
    print("\n" + "="*60)
    
    # Example 2: Recommendations for a user
    print("Example 2: Movie Recommendations")
    quick_predict(run_id=latest_run_id, user_id=645, top_k=10)
    
    print("\n" + "="*60)
    
    # Example 3: Load from registry (latest version)
    print("Example 3: Load from Model Registry")
    quick_predict(run_id=None, user_id=8648, top_k=5)
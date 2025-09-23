import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_model_and_predict():
    """Load the neural recommender model from MLflow and make sample predictions"""
    
    # Your specific run ID from the training
    run_id = "ef7c3a3244654413b979af93d39253e9"
    
    print("Loading trained neural recommender model from MLflow...")
    print(f"Run ID: {run_id}")
    
    # Load model from MLflow
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and prepare the same data to recreate encoders
    print("\nRecreating encoders from original dataset...")
    df = pd.read_csv("movies_ratings_cleaned.csv")
    
    # Sample the same way as training (150k rows)
    df = df.sample(n=150000, random_state=42)
    
    # Handle column mapping (same as training)
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
    
    # Create the same encoders as training
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    user_encoder.fit(df['userId'])
    movie_encoder.fit(df['movieId'])
    
    print(f"✅ Encoders ready: {len(user_encoder.classes_)} users, {len(movie_encoder.classes_)} movies")
    
    # Get some sample user and movie IDs for testing
    sample_users = df['userId'].unique()[:5]
    sample_movies = df['movieId'].unique()[:5]
    
    print(f"\nSample Users: {sample_users}")
    print(f"Sample Movies: {sample_movies}")
    
    # Make predictions
    print("\n" + "="*60)
    print("MAKING SAMPLE PREDICTIONS")
    print("="*60)
    
    model.eval()
    
    for i, user_id in enumerate(sample_users):
        print(f"\n--- Predictions for User {user_id} ---")
        
        for j, movie_id in enumerate(sample_movies):
            try:
                # Encode the IDs
                user_encoded = user_encoder.transform([user_id])[0]
                movie_encoded = movie_encoder.transform([movie_id])[0]
                
                # Convert to tensors
                user_tensor = torch.LongTensor([user_encoded])
                movie_tensor = torch.LongTensor([movie_encoded])
                
                # Make prediction
                with torch.no_grad():
                    prediction = model(user_tensor, movie_tensor)
                    predicted_rating = prediction.item()
                
                # Find actual rating if it exists
                actual_rating = df[(df['userId'] == user_id) & (df['movieId'] == movie_id)]['rating']
                actual_str = f"(Actual: {actual_rating.iloc[0]:.1f})" if len(actual_rating) > 0 else "(No actual rating)"
                
                print(f"  Movie {movie_id}: {predicted_rating:.2f} {actual_str}")
                
            except Exception as e:
                print(f"  Movie {movie_id}: Error - {e}")
        
        if i >= 2:  # Limit to first 3 users for demo
            break
    
    # Generate top recommendations for a user
    print(f"\n" + "="*60)
    print("TOP 10 MOVIE RECOMMENDATIONS")
    print("="*60)
    
    target_user = sample_users[0]
    print(f"Generating recommendations for User {target_user}...")
    
    # Get all movies
    all_movies = movie_encoder.classes_
    recommendations = []
    
    try:
        user_encoded = user_encoder.transform([target_user])[0]
        user_tensor = torch.LongTensor([user_encoded] * len(all_movies))
        movie_tensors = torch.LongTensor(range(len(all_movies)))
        
        with torch.no_grad():
            predictions = model(user_tensor, movie_tensors)
        
        # Create recommendations list
        for i, movie_id in enumerate(all_movies):
            recommendations.append((movie_id, predictions[i].item()))
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 recommended movies for User {target_user}:")
        for i, (movie_id, rating) in enumerate(recommendations[:10], 1):
            print(f"  {i:2d}. Movie {movie_id}: {rating:.2f}")
            
    except Exception as e:
        print(f"Error generating recommendations: {e}")
    
    print(f"\n" + "="*60)
    print("MODEL INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Show model info
    print(f"\nModel Information:")
    print(f"- MLflow Run ID: {run_id}")
    print(f"- Model Type: Neural Collaborative Filtering")
    print(f"- Users in training: {len(user_encoder.classes_):,}")
    print(f"- Movies in training: {len(movie_encoder.classes_):,}")
    print(f"- Training samples: {len(df):,}")

if __name__ == "__main__":
    load_model_and_predict()
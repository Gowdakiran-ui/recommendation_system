import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ModelInference:
    """Class to handle model inference using MLflow"""
    
    def __init__(self, model_name="neural_recommender", run_id=None):
        """
        Initialize the inference class
        
        Args:
            model_name: Name of the registered model in MLflow
            run_id: Specific run ID to load model from (optional)
        """
        self.model_name = model_name
        self.run_id = run_id
        self.model = None
        self.user_encoder = None
        self.movie_encoder = None
        
    def load_model_from_registry(self, version="latest"):
        """Load model from MLflow model registry"""
        try:
            model_uri = f"models:/{self.model_name}/{version}"
            print(f"Loading model from registry: {model_uri}")
            self.model = mlflow.pytorch.load_model(model_uri)
            print("Model loaded successfully from registry!")
            return True
        except Exception as e:
            print(f"Failed to load from registry: {e}")
            return False
    
    def load_model_from_run(self, run_id=None):
        """Load model from a specific MLflow run"""
        if run_id is None:
            run_id = self.run_id
        
        if run_id is None:
            print("No run ID provided!")
            return False
            
        try:
            model_uri = f"runs:/{run_id}/model"
            print(f"Loading model from run: {model_uri}")
            self.model = mlflow.pytorch.load_model(model_uri)
            print("Model loaded successfully from run!")
            return True
        except Exception as e:
            print(f"Failed to load from run: {e}")
            return False
    
    def prepare_encoders(self, original_data_path="movies_ratings_cleaned.csv", sample_size=150000):
        """
        Prepare the same encoders used during training
        Note: In production, you should save and load the actual encoders
        """
        print("Preparing encoders (recreating from original data)...")
        
        # Load and sample the same way as training
        df = pd.read_csv(original_data_path)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Handle column names (same logic as training)
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
        
        # Create encoders (same as training)
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        self.user_encoder.fit(df['userId'])
        self.movie_encoder.fit(df['movieId'])
        
        print(f"Encoders prepared for {len(self.user_encoder.classes_)} users and {len(self.movie_encoder.classes_)} movies")
        return True
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a single user-movie pair
        
        Args:
            user_id: Original user ID
            movie_id: Original movie ID
            
        Returns:
            Predicted rating or None if user/movie not in training data
        """
        if self.model is None:
            print("Model not loaded! Please load model first.")
            return None
        
        if self.user_encoder is None or self.movie_encoder is None:
            print("Encoders not prepared! Please prepare encoders first.")
            return None
        
        try:
            # Encode the IDs
            user_encoded = self.user_encoder.transform([user_id])[0]
            movie_encoded = self.movie_encoder.transform([movie_id])[0]
            
            # Convert to tensors
            user_tensor = torch.LongTensor([user_encoded])
            movie_tensor = torch.LongTensor([movie_encoded])
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(user_tensor, movie_tensor)
                return prediction.item()
                
        except ValueError as e:
            print(f"User ID {user_id} or Movie ID {movie_id} not found in training data")
            return None
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, user_movie_pairs):
        """
        Predict ratings for multiple user-movie pairs
        
        Args:
            user_movie_pairs: List of tuples [(user_id, movie_id), ...]
            
        Returns:
            List of predictions
        """
        predictions = []
        for user_id, movie_id in user_movie_pairs:
            pred = self.predict_rating(user_id, movie_id)
            predictions.append(pred)
        return predictions
    
    def recommend_movies_for_user(self, user_id, top_k=10):
        """
        Recommend top-k movies for a specific user
        
        Args:
            user_id: Original user ID
            top_k: Number of recommendations to return
            
        Returns:
            List of tuples [(movie_id, predicted_rating), ...]
        """
        if self.model is None or self.user_encoder is None or self.movie_encoder is None:
            print("Model or encoders not loaded!")
            return []
        
        try:
            # Check if user exists in training data
            user_encoded = self.user_encoder.transform([user_id])[0]
        except ValueError:
            print(f"User ID {user_id} not found in training data")
            return []
        
        # Get all movie IDs
        all_movies = self.movie_encoder.classes_
        predictions = []
        
        print(f"Generating predictions for {len(all_movies)} movies...")
        
        # Predict for all movies
        user_tensor = torch.LongTensor([user_encoded] * len(all_movies))
        movie_tensors = torch.LongTensor(range(len(all_movies)))
        
        self.model.eval()
        with torch.no_grad():
            batch_predictions = self.model(user_tensor, movie_tensors)
            
        # Create list of (movie_id, prediction) pairs
        for i, movie_id in enumerate(all_movies):
            predictions.append((movie_id, batch_predictions[i].item()))
        
        # Sort by predicted rating and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

def main():
    """Demo script showing how to use the inference class"""
    
    # Initialize inference class with the run ID from training
    # Replace this with your actual run ID
    run_id = "f39cf7a3a57743d0a71fdea37667a91a"  # Your run ID
    
    inference = ModelInference(model_name="neural_recommender", run_id=run_id)
    
    print("=== MLflow Model Inference Demo ===\n")
    
    # Try loading from registry first, fallback to run if needed
    print("1. Loading model...")
    if not inference.load_model_from_registry():
        print("Registry loading failed, trying run-based loading...")
        if not inference.load_model_from_run():
            print("Failed to load model from both registry and run!")
            return
    
    # Prepare encoders
    print("\n2. Preparing encoders...")
    if not inference.prepare_encoders():
        print("Failed to prepare encoders!")
        return
    
    # Example 1: Single prediction
    print("\n3. Single Prediction Example:")
    print("-" * 40)
    
    # You'll need to replace these with actual user/movie IDs from your dataset
    sample_user_id = 1  # Replace with actual user ID
    sample_movie_id = 1  # Replace with actual movie ID
    
    prediction = inference.predict_rating(sample_user_id, sample_movie_id)
    if prediction is not None:
        print(f"Predicted rating for User {sample_user_id}, Movie {sample_movie_id}: {prediction:.2f}")
    else:
        print("Could not make prediction (user/movie not in training data)")
    
    # Example 2: Batch predictions
    print("\n4. Batch Prediction Example:")
    print("-" * 40)
    
    # Sample user-movie pairs (replace with actual IDs)
    sample_pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    batch_predictions = inference.predict_batch(sample_pairs)
    
    for (user_id, movie_id), pred in zip(sample_pairs, batch_predictions):
        if pred is not None:
            print(f"User {user_id}, Movie {movie_id}: {pred:.2f}")
        else:
            print(f"User {user_id}, Movie {movie_id}: No prediction available")
    
    # Example 3: Movie recommendations
    print("\n5. Movie Recommendations Example:")
    print("-" * 40)
    
    sample_user_for_recs = 1  # Replace with actual user ID
    recommendations = inference.recommend_movies_for_user(sample_user_for_recs, top_k=5)
    
    if recommendations:
        print(f"Top 5 movie recommendations for User {sample_user_for_recs}:")
        for i, (movie_id, rating) in enumerate(recommendations, 1):
            print(f"  {i}. Movie {movie_id}: {rating:.2f}")
    else:
        print("Could not generate recommendations")
    
    print("\n=== Demo completed! ===")

if __name__ == "__main__":
    main()
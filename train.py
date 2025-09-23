import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.pytorch
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class MovieLensDataset(Dataset):
    """Custom Dataset for MovieLens data"""
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

class NeuralRecommender(nn.Module):
    """Neural Network based Recommender System"""
    def __init__(self, num_users, num_movies, embedding_dim=50, hidden_dim1=128, hidden_dim2=64):
        super().__init__()
        
        # User and Movie embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Neural network layers
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Initialize embeddings with small random weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
    
    def forward(self, user_ids, movie_ids):
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        
        # Pass through neural network
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x.squeeze()

def load_and_preprocess_data(file_path, sample_size=150000):
    """Load and preprocess the MovieLens dataset"""
    print("Loading dataset...")
    
    # Load the full dataset
    df = pd.read_csv(file_path)
    print(f"Original dataset size: {len(df):,} rows")
    
    # Randomly sample the specified number of rows
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size:,} rows from the dataset")
    else:
        print(f"Using all {len(df):,} rows (less than sample size)")
    
    # Assume the CSV has columns: userId, movieId, rating
    # Adjust column names if different
    expected_columns = ['userId', 'movieId', 'rating']
    if not all(col in df.columns for col in expected_columns):
        print("Available columns:", df.columns.tolist())
        # Try to map common variations
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
            print("Mapped columns:", column_mapping)
    
    # Select only the required columns
    df = df[['userId', 'movieId', 'rating']].copy()
    
    # Remove any rows with missing values
    df = df.dropna()
    print(f"After removing missing values: {len(df):,} rows")
    
    # Encode user and movie IDs
    print("Encoding user and movie IDs...")
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    df['userId_encoded'] = user_encoder.fit_transform(df['userId'])
    df['movieId_encoded'] = movie_encoder.fit_transform(df['movieId'])
    
    num_users = len(df['userId_encoded'].unique())
    num_movies = len(df['movieId_encoded'].unique())
    
    print(f"Number of unique users: {num_users:,}")
    print(f"Number of unique movies: {num_movies:,}")
    print(f"Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")
    
    return df, num_users, num_movies, user_encoder, movie_encoder

def create_data_loaders(df, batch_size=512, test_split=0.2):
    """Create PyTorch DataLoaders for training and testing"""
    
    # Create dataset
    dataset = MovieLensDataset(
        df['userId_encoded'].values,
        df['movieId_encoded'].values,
        df['rating'].values
    )
    
    # Split into train and test
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Testing samples: {len(test_dataset):,}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=25, lr=0.001):
    """Train the neural recommender model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for user_ids, movie_ids, ratings in train_loader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")
        
        # Log training loss to MLflow
        mlflow.log_metric("training_loss", avg_loss, step=epoch)
    
    # Calculate test RMSE
    model.eval()
    test_predictions = []
    test_actuals = []
    
    with torch.no_grad():
        for user_ids, movie_ids, ratings in test_loader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            predictions = model(user_ids, movie_ids)
            
            test_predictions.extend(predictions.cpu().numpy())
            test_actuals.extend(ratings.cpu().numpy())
    
    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
    print(f"\nFinal Test RMSE: {test_rmse:.4f}")
    
    return model, test_rmse, train_losses

def main():
    """Main function to run the complete pipeline"""
    
    # Hyperparameters
    SAMPLE_SIZE = 150000
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 25
    EMBEDDING_DIM = 50
    HIDDEN_DIM1 = 128
    HIDDEN_DIM2 = 64
    
    # Start MLflow run
    with mlflow.start_run():
        
        # Log hyperparameters
        mlflow.log_params({
            "sample_size": SAMPLE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim1": HIDDEN_DIM1,
            "hidden_dim2": HIDDEN_DIM2
        })
        
        # Load and preprocess data
        df, num_users, num_movies, user_encoder, movie_encoder = load_and_preprocess_data(
            "movies_ratings_cleaned.csv", 
            sample_size=SAMPLE_SIZE
        )
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(df, batch_size=BATCH_SIZE)
        
        # Create model
        model = NeuralRecommender(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim1=HIDDEN_DIM1,
            hidden_dim2=HIDDEN_DIM2
        )
        
        print(f"\nModel architecture:")
        print(f"Users: {num_users}, Movies: {num_movies}")
        print(f"Embedding dimension: {EMBEDDING_DIM}")
        print(f"Hidden layers: {HIDDEN_DIM1} -> {HIDDEN_DIM2} -> 1")
        
        # Train model
        print(f"\nStarting training for {EPOCHS} epochs...")
        trained_model, test_rmse, train_losses = train_model(
            model, train_loader, test_loader, 
            epochs=EPOCHS, lr=LEARNING_RATE
        )
        
        # Log final metrics
        mlflow.log_metrics({
            "final_test_rmse": float(test_rmse),
            "num_users": float(num_users),
            "num_movies": float(num_movies),
            "dataset_size": float(len(df))
        })
        
        # Save model to MLflow
        mlflow.pytorch.log_model(
            trained_model, 
            "model",
            registered_model_name="neural_recommender"
        )
        
        print(f"\nTraining completed!")
        print(f"Model saved to MLflow as 'neural_recommender'")
        print(f"Final Test RMSE: {test_rmse:.4f}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()

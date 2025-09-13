import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ===============================
# 1. Dataset & Preprocessing
# ===============================
class MovieDataset(Dataset):
    def __init__(self, df, user_encoder, movie_encoder):
        self.user_ids = torch.tensor(user_encoder.transform(df['userId']), dtype=torch.long)
        self.movie_ids = torch.tensor(movie_encoder.transform(df['movieId']), dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

# ===============================
# 2. Neural Collaborative Filtering Model
# ===============================
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, movie_ids):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        x = torch.cat([user_embed, movie_embed], dim=-1)
        return self.fc_layers(x).squeeze()

# ===============================
# 3. Training Function
# ===============================
def train_model(df, model_path="best_ncf_model.pth", batch_size=256, epochs=5, lr=0.001, embedding_dim=64):
    # Encode users and movies
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df['userId_enc'] = user_encoder.fit_transform(df['userId'])
    df['movieId_enc'] = movie_encoder.fit_transform(df['movieId'])

    num_users = df['userId_enc'].nunique()
    num_movies = df['movieId_enc'].nunique()

    # Train-test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = MovieDataset(train_df, user_encoder, movie_encoder)
    val_dataset = MovieDataset(val_df, user_encoder, movie_encoder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users, num_movies, embedding_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for user_ids, movie_ids, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)

            optimizer.zero_grad()
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user_ids, movie_ids, ratings in val_loader:
                user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
                outputs = model(user_ids, movie_ids)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "user_encoder": user_encoder,
                "movie_encoder": movie_encoder
            }, model_path)
            print(f"✅ Model saved at {model_path}")

    return model

# ===============================
# 4. Run Training
# ===============================
if __name__ == "__main__":
    # Load your cleaned dataset (CSV file)
    df = pd.read_csv("cleaned_movies.csv")  # <-- replace with your dataset file

    trained_model = train_model(df, model_path="best_ncf_model.pth", epochs=5)

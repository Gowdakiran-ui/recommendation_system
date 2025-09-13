# app.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn

# Define the same NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc_layers(x)

# Init Flask
app = Flask(__name__)

# Define model params (must match training)
num_users = 200000  
num_items = 30000   
embedding_dim = 32
hidden_dim = 64

# Load model
model = NCF(num_users, num_items, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("best_ncf_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_id = torch.tensor([data["user_id"]])
    item_id = torch.tensor([data["item_id"]])

    with torch.no_grad():
        prediction = model(user_id, item_id).item()

    return jsonify({"user_id": data["user_id"], 
                    "item_id": data["item_id"], 
                    "predicted_rating": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)

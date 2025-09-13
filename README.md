

# 🎬 Neural Collaborative Filtering (NCF) Recommendation System

This project implements a **Recommendation System** using **Neural Collaborative Filtering (NCF)** in PyTorch.
The trained model is saved as a `.pth` file and can be loaded inside a **Flask app** to serve predictions via an API.

---

## 🚀 Features

* Train and save an NCF-based recommendation model.
* Load the trained model (`.pth`) and use it for predictions.
* Flask API for serving predictions (`user_id`, `item_id` → `predicted_rating`).
* Easy integration with other apps (frontend, dashboards, or microservices).

---

## 🛠 Tech Stack

* **Python 3.10+**
* **PyTorch** (model training & inference)
* **Flask** (API serving)
* **scikit-learn**, **pandas**, **numpy** (for dataset preprocessing & splitting)

---

## 📂 Project Structure

```
recomendations/
│── app.py                  # Flask API to serve predictions
│── load_model.py           # Standalone script to load & test the model
│── best_ncf_model.pth      # Saved trained model (weights only)
│── requirements.txt        # Project dependencies
│── README.md               # Documentation
```

---

## ⚡️ Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the model directly

```bash
python load_model.py
```

Example output:

```
Predicted rating for user 1, item 17: 3.71
```

### 3. Run the Flask API

```bash
python app.py
```

Flask server will start at: `http://127.0.0.1:5000`

---

## 📡 API Endpoints

### **POST** `/predict`

Predicts a rating for a given user and item.

**Request body:**

```json
{
  "user_id": 1,
  "item_id": 17
}
```

**Response:**

```json
{
  "user_id": 1,
  "item_id": 17,
  "predicted_rating": 3.71
}
```

---

## 🧠 Model Details

* **Architecture:** Neural Collaborative Filtering (NCF)
* **Embeddings:** User & Item embeddings combined
* **Hidden Layers:** Fully connected layers with ReLU activation
* **Output:** Predicted rating (continuous value)

The model was trained on user-item interaction data and saved as `best_ncf_model.pth`.

---

## 🎯 Next Steps

* Add a **frontend (HTML form)** to take `user_id` and `item_id`.
* Extend API for **top-N recommendations** per user.
* Deploy to **Heroku / Render / AWS / GCP** for production use.

---

🔥 With this setup, you can train, save, and serve your recommendation system in just a few steps!

---




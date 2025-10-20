⸻


# 🏍️ Rider Intention Prediction System

**Author:** Shavir Sejal Morar  
**Institution:** University of Johannesburg – BSc Honours in Computer Science  
**Version:** 1.0.3  
**Deployed:** [Render Web App](https://rider-intention-prediction-system.onrender.com)

---

## 📘 Overview

The **Rider Intention Prediction (RIP)** system is a deep learning-based framework designed to classify a motorcyclist’s riding intention (e.g., turning, accelerating, braking) from sequential video features.  
The model integrates a **feature extraction backbone** with a **Bidirectional LSTM classifier**, enabling real-time prediction of rider intentions using temporal sequences.

This system aims to enhance **two-wheeler safety** by predicting intentions early enough for Advanced Rider Assistance Systems (ARAS) to respond appropriately.

---

## 🧠 Project Architecture

Input Video  →  ResNet Feature Extractor  →  Sequence of Frame Features (.npy)
↓
LSTMFeatureClassifier (PyTorch)
↓
Predicted Rider Intention (Label + Confidence)

### Core Components
- **Feature Extraction:** Each video clip is processed to extract frame-wise feature embeddings (2048-dimensional).
- **Sequence Modelling:** An **LSTM** network captures spatio-temporal dependencies within the feature sequences.
- **Classification:** The model outputs probabilities for six possible rider intentions.
- **Web Deployment:** FastAPI backend with a responsive HTML/JS interface for easy testing.

---

## 🧩 Directory Structure

rider-intent/
│
├── src/
│   ├── server/
│   │   ├── main.py              # FastAPI server (API + UI)
│   │   └── static/index.html    # Frontend interface
│   │
│   ├── features/
│   │   ├── lstm_classifier.py   # PyTorch LSTM model
│   │   ├── dataset.py           # Dataset + preprocessing
│   │   └── utils.py             # Helper functions (if any)
│   │
│   ├── training/
│   │   ├── train_features.py    # Model training script
│   │   └── evaluate.py          # Evaluation + confusion matrix
│   │
│   └── dist/
│       └── model.pt             # Trained model weights
│
├── config.yaml                  # Dataset + training configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shivs-m/Rider-Intention-Prediction-System.git
cd Rider-Intention-Prediction-System

2️⃣ Create and Activate a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # Mac / Linux
venv\Scripts\activate     # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt


⸻

🧪 Local Development

Run the FastAPI Server

uvicorn server.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload

Then open:
	•	Web UI → http://127.0.0.1:8000
	•	API Docs → http://127.0.0.1:8000/docs
	•	Health Check → http://127.0.0.1:8000/healthz

⸻

🚀 Deployment (Render)

This app is fully compatible with Render’s free tier.

Start Command

uvicorn server.main:app --app-dir src --host 0.0.0.0 --port $PORT

Health Check URL

/healthz

Make sure your dist/model.pt file is committed to the repository before deployment.

⸻

📊 Model Training Results

Metric	Epoch 1	Epoch 3	Epoch 5
Loss	1.73	1.38	1.13
Accuracy	0.31	0.31	0.35
Macro F1	0.23	0.23	0.29

The model achieved consistent improvements across epochs, indicating stable training and generalisation.

⸻

🧠 Key Technologies

Category	Tools / Frameworks
Backend	FastAPI, Uvicorn
ML Framework	PyTorch
Frontend	HTML, CSS, JavaScript
Deployment	Render Cloud
Version Control	Git + GitHub
Dataset Handling	NumPy, YAML


⸻

🧩 Example API Usage

POST /predict_feature

Input:
Upload a .npy file containing a sequence of frame features (shape = [T, D]).

Response Example:

{
  "label": "turn_left",
  "confidence": 0.87,
  "probs": [0.87, 0.03, 0.02, 0.04, 0.02, 0.02],
  "classes": ["turn_left", "turn_right", "accelerate", "brake", "idle", "lane_change"]
}


⸻

💡 Future Work
	•	Integrate real-time video inference (upload raw video → auto-extract features).
	•	Expand dataset for diverse weather, lighting, and rider conditions.
	•	Deploy as a mobile-friendly PWA for traffic-safety research pilots.

⸻

👨‍💻 Author

Shavir Sejal Morar
📧 222028506@student.uj.ac.za
🎓 University of Johannesburg
🧩 BSc Honours in Computer Science

⸻

📚 References
	•	Graves, A. (2013). Speech recognition with deep recurrent neural networks. IEEE ICASSP.
	•	Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.
	•	Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
	•	Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.

⸻

🏁 License

This project is for academic research and educational purposes only under the University of Johannesburg Honours Program.
For external use, please credit the original author.

⸻

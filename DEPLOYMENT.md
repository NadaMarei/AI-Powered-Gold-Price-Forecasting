# GoldAI Pro - Deployment Guide

## 🚀 Quick Deployment to Render

### Option 1: One-Click Deploy (Recommended)

1. Push this repository to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click **"New"** → **"Web Service"**
4. Connect your GitHub repository
5. Render will automatically detect the configuration from `render.yaml`
6. Click **"Create Web Service"**

### Option 2: Manual Configuration

1. Create a new Web Service on Render
2. Connect your repository
3. Configure:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app.main:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Health Check Path**: `/health`

## 📁 Project Structure for Deployment

```
gold/
├── app/                    # Web Application
│   ├── main.py            # Flask Application
│   ├── templates/         # HTML Templates
│   │   └── index.html     # Main Dashboard
│   └── static/            # Static Assets
│       ├── styles.css     # Styles
│       └── app.js         # Frontend JavaScript
├── src/                   # ML Model Code
├── checkpoints/           # Trained Models
├── Procfile              # Render Process File
├── render.yaml           # Render Blueprint
├── runtime.txt           # Python Version
├── requirements.txt      # Dependencies
├── gunicorn.conf.py      # Production Server Config
└── wsgi.py               # WSGI Entry Point
```

## 🔧 Environment Variables

Set these in your Render dashboard:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port (auto-set by Render) |
| `FLASK_ENV` | `production` | Flask environment |
| `MODEL_PATH` | `checkpoints/best_model.pt` | Path to trained model |

## 🧪 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m app.main

# Access at http://localhost:5000
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/current-price` | GET | Current gold price |
| `/api/predict` | GET | AI prediction |
| `/api/historical` | GET | Historical data |
| `/api/model-metrics` | GET | Model performance |
| `/api/feature-importance` | GET | XAI insights |
| `/health` | GET | Health check |

## 🎨 Features

- **Real-time Gold Prices**: Live data from Yahoo Finance
- **AI Predictions**: GRU-based deep learning forecasts
- **Interactive Charts**: Responsive Chart.js visualizations
- **XAI Dashboard**: Explainable AI insights
- **Responsive Design**: Works on desktop and mobile
- **Dark Theme**: Professional, modern aesthetics

## 🔒 Security Notes

- The app runs in demo mode if no trained model is present
- Data is fetched from public APIs (Yahoo Finance)
- No user data is stored or collected

## 📈 Performance

- Optimized for Render's free tier
- 2 Gunicorn workers
- 120s timeout for model inference
- Health checks enabled

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `checkpoints/best_model.pt` exists
- Check PyTorch version compatibility
- Model will run in demo mode if unavailable

### Slow Startup
- First request may be slow due to data fetching
- Subsequent requests are cached

### Port Issues
- Render sets PORT automatically
- Local: defaults to 5000

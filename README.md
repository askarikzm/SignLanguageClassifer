# Hand Gesture Recognition App

A Flask-based hand gesture recognition application using MediaPipe and XGBoost.

## Deployment to Render

### Prerequisites
- Git installed
- GitHub/GitLab account
- Render account (free tier available)

### Steps to Deploy

1. **Initialize Git Repository (if not already done)**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub**
   ```bash
   # Create a new repository on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the Dockerfile
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

### Alternative: Manual Render Setup

If `render.yaml` doesn't auto-configure:
- **Environment**: Docker
- **Build Command**: (leave blank - uses Dockerfile)
- **Start Command**: (leave blank - uses Dockerfile CMD)
- **Plan**: Free

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python api.py
```

## Files
- `api.py` - Main Flask API
- `app.py` - Alternative Flask app with UI
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `*.pkl` - ML model files (XGBoost model, scaler, class mappings)

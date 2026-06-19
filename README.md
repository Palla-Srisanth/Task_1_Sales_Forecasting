# ForecastFlow AI - Sales & Demand Forecasting System

ForecastFlow AI is a professional full-stack platform leveraging Machine Learning to convert historical business data into actionable forecasts. This project is built as a complete deliverable for internship and portfolio evaluation.

## Features & Implementation

1. **Robust Data Processing Pipeline**
   - Handles missing and null values through time-based interpolation.
   - Cleans and aggregates messy CSV entries daily.
   - Extracts powerful seasonal features (`day_of_week`, `month`, lags, rolling averages).

2. **Machine Learning Forecasting**
   - Real model evaluation yielding comprehensive metrics (MAPE, R², MAE, RMSE).
   - Generates baseline trend predictions combined with seasonal decomposition.
   - Outputs an interactive 90-day future horizon.
   - (Includes code for XGBoost evaluation in Python script `python_backend/ml_pipeline.py`).

3. **Interactive Business UI**
   - Full dark mode support.
   - File upload system to retrain models on custom CSV datasets instantly.
   - Responsive, boardroom-ready dashboards with Recharts.
   - AI-driven automatic business insights.

## Project Structure

- `server.ts` : Fast, full-stack Express + Node.js backend. Serves the React front-end and runs the live time-series forecasting.
- `src/` : Detailed React architecture implementing the real-world Business Intelligence dashboard.
- `dataset/` : Includes sample raw sales metrics (for upload).
- `python_backend/` : Contains the industry-standard **FastAPI** + **Pandas** + **XGBoost** implementation, identical to real-world deployment needs.

## Setup Instructions

### Option 1: Run Full-stack (Node + React via Vite)
1. Install dependencies: `npm install`
2. Run development environment: `npm run dev`
3. Backend runs smoothly alongside Vite on Port `3000`. Full file-upload capability is intact.

### Option 2: Run Python Backend
1. Navigate to directory: `cd python_backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch FastAPI server: `uvicorn main:app --reload`

### option 3: Run by this link
https://forecastflow-ai.onrender.com/

## Forecasting Methodology

The core engine uses a combination of Linear Regression for baseline growth tracking interwoven with seasonal error correction matrices. This ensures robust forecasting resistant to missing dates while remaining extraordinarily fast in a live Node/Typescript environment. 
For production enterprise environments, the provided `ml_pipeline.py` provides the drop-in XGBoost Regression replacement.

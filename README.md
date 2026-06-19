# Task 1 - ForecastFlow AI: Sales & Demand Forecasting System

## Overview

ForecastFlow AI is a full-stack Sales & Demand Forecasting platform designed to transform historical business data into actionable forecasts and business insights. The system combines data processing, machine learning forecasting, interactive visualizations, and business intelligence features to help organizations make data-driven decisions.

## Key Features

### Data Processing Pipeline

* Handles missing and null values using interpolation techniques
* Cleans and aggregates raw sales datasets
* Performs feature engineering using:

  * Day of Week
  * Month
  * Lag Features
  * Rolling Averages
* Supports custom dataset uploads

### Machine Learning Forecasting

* Generates sales and demand forecasts for future periods
* Model evaluation using:

  * MAPE (Mean Absolute Percentage Error)
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * R² Score
* Seasonal trend analysis and forecasting
* Includes XGBoost-based forecasting pipeline

### Interactive Dashboard

* Modern responsive UI
* Dark mode interface
* Forecast visualization charts
* Category-wise demand analysis
* AI-generated business insights
* Export report functionality

## Technology Stack

### Frontend

* React
* TypeScript
* Vite
* Recharts

### Backend

* Node.js
* Express.js

### Machine Learning

* Python
* FastAPI
* Pandas
* NumPy
* Scikit-learn
* XGBoost

## Project Structure

```text
ForecastFlow-AI/
│
├── src/                 # React Frontend
├── python_backend/      # FastAPI & ML Pipeline
├── dataset/             # Sample Datasets
├── server.ts            # Node.js Backend
├── package.json         # Project Dependencies
├── tsconfig.json        # TypeScript Configuration
└── README.md
```

## Installation & Setup

### Clone Repository

```bash
git clone <repository-url>
cd ForecastFlow-AI
```

### Install Dependencies

```bash
npm install
```

### Run Application

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

### Run Python ML Backend

```bash
cd python_backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Business Value

This system helps organizations:

* Forecast future sales trends
* Optimize inventory management
* Improve demand planning
* Identify seasonal patterns
* Support data-driven decision making
* Generate actionable business insights

## Author

**Srisanth Palla**

B.Tech CSE (AI & ML)
Pragati Engineering College

## Task Submission

This repository was submitted as **Task 1 - Sales Forecasting & Demand Intelligence System**.

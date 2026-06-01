# Nepride
its a ride sharing management system 
# NepRide AI — ML Service Documentation

## Overview

The ML service is a **FastAPI** application that provides two prediction endpoints
and an analytics endpoint. It uses **scikit-learn** models trained on synthetic
Kathmandu ride data.



---

## Models

### 1. Demand Prediction Model
**Algorithm:** Random Forest Classifier
**Purpose:** Predict ride demand level for a given zone/time

**Input Features:**
| Feature      | Type    | Description                        |
|--------------|---------|------------------------------------|
| hour         | int     | Hour of day (0-23)                 |
| day_of_week  | int     | 0=Monday, 6=Sunday                 |
| month        | int     | 1-12                               |
| zone_encoded | int     | Encoded zone (Thamel=0, Patan=1..) |
| is_weekend   | int     | 0 or 1                             |
| is_holiday   | int     | 0 or 1                             |

**Output:**
```json
{
  "zone": "thamel",
  "hour": 8,
  "demand_level": "high",
  "demand_score": 0.82,
  "surge_multiplier": 1.6,
  "estimated_wait_minutes": 3
}
```

**Demand Levels:** `low` | `medium` | `high` | `very_high`

---

### 2. Traffic Prediction Model
**Algorithm:** Gradient Boosting Regressor
**Purpose:** Predict traffic congestion level at a given location/time

**Input Features:**
| Feature      | Type    | Description                        |
|--------------|---------|------------------------------------|
| latitude     | float   | Location latitude                  |
| longitude    | float   | Location longitude                 |
| hour         | int     | Hour of day                        |
| day_of_week  | int     | Day of week                        |
| is_peak_hour | int     | 0 or 1 (7-9am, 5-7pm)             |

**Output:**
```json
{
  "location": { "lat": 27.7172, "lon": 85.3240 },
  "traffic_level": "heavy",
  "congestion_score": 0.74,
  "estimated_delay_minutes": 12,
  "recommendation": "Consider alternative route via Ring Road"
}
```

**Traffic Levels:** `free` | `moderate` | `heavy` | `standstill`

---

## API Endpoints

### Health Check
```
GET /health
Response: { "status": "ok", "models_loaded": true }
```

### Demand Prediction
```
GET /predict/demand?zone=thamel&hour=8&day=Monday

Response:
{
  "success": true,
  "data": {
    "zone": "thamel",
    "hour": 8,
    "day": "Monday",
    "demand_level": "high",
    "demand_score": 0.82,
    "surge_multiplier": 1.6,
    "estimated_wait_minutes": 3,
    "prediction_confidence": 0.91
  }
}
```

### Traffic Prediction
```
GET /predict/traffic?lat=27.7172&lon=85.3240

Response:
{
  "success": true,
  "data": {
    "location": { "lat": 27.7172, "lon": 85.3240 },
    "traffic_level": "heavy",
    "congestion_score": 0.74,
    "estimated_delay_minutes": 12,
    "recommendation": "Consider alternative route via Ring Road"
  }
}
```

### Analytics
```
GET /analytics

Response:
{
  "success": true,
  "data": {
    "peak_hours": [8, 9, 17, 18, 19],
    "busiest_zones": ["Thamel", "New Road", "Patan", "Bhaktapur"],
    "demand_by_hour": [...],
    "demand_by_day": [...],
    "avg_ride_duration": 22,
    "avg_fare_npr": 210
  }
}
```

---

## Kathmandu Zones

| Zone ID | Zone Name     | Coordinates (approx)      |
|---------|---------------|---------------------------|
| 0       | Thamel        | 27.7154, 85.3123          |
| 1       | New Road      | 27.7041, 85.3145          |
| 2       | Patan         | 27.6710, 85.3256          |
| 3       | Bhaktapur     | 27.6710, 85.4298          |
| 4       | Kirtipur      | 27.6783, 85.2798          |
| 5       | Boudha        | 27.7215, 85.3621          |
| 6       | Swayambhu     | 27.7149, 85.2904          |
| 7       | Baneshwor     | 27.6939, 85.3417          |
| 8       | Koteshwor     | 27.6782, 85.3567          |
| 9       | Kalanki       | 27.6977, 85.2820          |

---

## Peak Hours (Kathmandu)

```
Morning peak:  07:00 – 09:30  (office/school commute)
Lunch peak:    12:00 – 13:30  (lunch breaks)
Evening peak:  17:00 – 19:30  (return commute)
Night peak:    21:00 – 23:00  (restaurants/entertainment)
```

---

## Model Training

```bash
cd ml-service
python services/train.py

# Output:
# Training demand model...   Accuracy: 0.87
# Training traffic model...  R² Score: 0.81
# Models saved to saved_models/
```

---

## File Structure

```
ml-service/
├── app/
│   └── main.py              # FastAPI app factory
├── routers/
│   ├── predict.py           # /predict/* endpoints
│   └── analytics.py         # /analytics endpoint
├── models/
│   ├── demand_model.py      # RandomForest wrapper
│   └── traffic_model.py     # GradientBoosting wrapper
├── services/
│   ├── train.py             # Training pipeline
│   └── predict.py           # Prediction logic
├── data/
│   ├── raw/                 # Raw CSV datasets
│   └── processed/           # Cleaned datasets
├── notebooks/
│   ├── eda.ipynb            # Exploratory analysis
│   └── model_training.ipynb # Training notebook
├── saved_models/
│   ├── demand_model.pkl     # Trained demand model
│   └── traffic_model.pkl    # Trained traffic model
├── requirements.txt
└── README.md
```

---

## Requirements

```
fastapi==0.110.0
uvicorn==0.29.0
scikit-learn==1.4.2
pandas==2.2.1
numpy==1.26.4
joblib==1.3.2
python-dotenv==1.0.1
httpx==0.27.0
```

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install API dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

# Flask application
COPY app.py .

# Trained XGBoost model
COPY data/output/xgb_model.pkl \
     data/output/xgb_model.pkl

# Training encoder
COPY data/interim/encoder.pkl \
     data/interim/encoder.pkl

# Exact feature order used during training
COPY data/interim/feature_names.txt \
     data/interim/feature_names.txt

# Validation-selected threshold
COPY reports/threshold.txt \
     reports/threshold.txt

EXPOSE 5000

CMD ["python", "app.py"]
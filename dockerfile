# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything to /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port if your app runs a web service (optional for now)
EXPOSE 5000

# Run full pipeline sequentially
CMD ["bash", "-c", "python src/data_ingestion.py && python src/preprocessing.py && python src/model.py"]

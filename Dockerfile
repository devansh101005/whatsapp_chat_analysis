# Simple Docker image to run the WhatsApp Chat Analysis Streamlit app
FROM python:3.11-slim

# Keep Python output unbuffered so logs show up right away
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

# Copy the rest of the project
COPY . .

# Streamlit runs on port 8501
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

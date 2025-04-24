#!/bin/bash

# Step 1: Build the container 
gcloud builds submit --tag gcr.io/rag-final-456320/serving-rag-gemini

# Step 2: Deploy 
gcloud run deploy capitolinsight \
  --image gcr.io/rag-final-456320/serving-rag-gemini \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --update-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --port 8080 \
  --memory=4Gi \
  --min-instances=1 \
  --timeout=600 \
  --quiet

# Recommendation-System

The app is deployed in Google Cloud Run.

Scale it up/down using these commands:

gcloud run services update streamlit-app --min-instances=0

gcloud run services update streamlit-app   --memory=1Gi   --cpu=1
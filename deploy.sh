#!/bin/bash

# AI-Enhanced Financial Cleaner - Cloud Run Deployment Script
# Usage: ./deploy.sh [project-id] [region] [api-key]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="ai-financial-cleaner"
DEFAULT_REGION="us-central1"
DEFAULT_PROJECT=""

# Parse arguments
PROJECT_ID=${1:-$DEFAULT_PROJECT}
REGION=${2:-$DEFAULT_REGION}
ANTHROPIC_API_KEY=${3:-$ANTHROPIC_API_KEY}

echo -e "${BLUE}🚀 AI-Enhanced Financial Cleaner Deployment${NC}"
echo "=================================================="

# Validate inputs
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: Project ID required${NC}"
    echo "Usage: ./deploy.sh [project-id] [region] [api-key]"
    echo "Example: ./deploy.sh my-project us-central1 sk-ant-..."
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: No Anthropic API key provided. Service will run in mock mode.${NC}"
    echo "Set ANTHROPIC_API_KEY environment variable or pass as third argument for live AI."
fi

echo -e "${BLUE}📋 Deployment Configuration${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "API Key: ${ANTHROPIC_API_KEY:+Provided}${ANTHROPIC_API_KEY:-Not provided (mock mode)}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker not found${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Set project
echo -e "${BLUE}🔧 Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${BLUE}🔌 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy
echo -e "${BLUE}🏗️  Building and deploying to Cloud Run...${NC}"

if [ -n "$ANTHROPIC_API_KEY" ]; then
    # Deploy with API key
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY,FLASK_ENV=production,ENABLE_AI=true"
else
    # Deploy in mock mode
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production,ENABLE_AI=false"
fi

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo "=================================================="
echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
echo ""
echo -e "${BLUE}📖 Available Endpoints:${NC}"
echo "• GET  $SERVICE_URL/health - Health check"
echo "• GET  $SERVICE_URL/stats - Usage statistics"
echo "• POST $SERVICE_URL/process - Process financial data"
echo "• POST $SERVICE_URL/demo - Demo with sample data"
echo ""
echo -e "${BLUE}🧪 Test the deployment:${NC}"
echo "curl $SERVICE_URL/health"
echo ""
echo "curl -X POST $SERVICE_URL/demo"
echo ""
echo -e "${BLUE}📊 Monitor the service:${NC}"
echo "gcloud run services describe $SERVICE_NAME --region=$REGION"
echo ""
echo -e "${BLUE}🔧 Update environment variables:${NC}"
echo "gcloud run services update $SERVICE_NAME --region=$REGION --set-env-vars KEY=VALUE"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Remember to set your Anthropic API key for live AI:${NC}"
    echo "gcloud run services update $SERVICE_NAME --region=$REGION --set-env-vars ANTHROPIC_API_KEY=your-key-here,ENABLE_AI=true"
    echo ""
fi

echo -e "${GREEN}🎉 Your AI-Enhanced Financial Cleaner is now live!${NC}" 
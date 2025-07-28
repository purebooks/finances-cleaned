# 🚀 Deployment Guide - AI-Enhanced Financial Cleaner

## 📋 **Prerequisites**

Before deploying, ensure you have:

1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI** installed and configured
3. **Docker** installed (for local testing)
4. **Anthropic API Key** (optional, service works in mock mode without it)

## 🛠️ **Setup Instructions**

### **1. Clone/Setup Repository**

```bash
# If you have the files locally, navigate to the deployment directory
cd ai-financial-cleaner-deploy

# Or create a new GitHub repository and push these files
git init
git add .
git commit -m "Initial commit - AI-Enhanced Financial Cleaner"
git remote add origin https://github.com/yourusername/ai-financial-cleaner.git
git push -u origin main
```

### **2. Install Google Cloud CLI**

```bash
# macOS
brew install google-cloud-sdk

# Ubuntu/Debian
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli

# Windows
# Download installer from: https://cloud.google.com/sdk/docs/install
```

### **3. Configure Google Cloud**

```bash
# Login to Google Cloud
gcloud auth login

# Set your project (create one if needed)
gcloud config set project YOUR_PROJECT_ID

# Enable billing for your project (required for Cloud Run)
# Visit: https://console.cloud.google.com/billing
```

## 🚀 **Deployment Options**

### **Option 1: Automated Deployment (Recommended)**

Use the included deployment script:

```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy with Anthropic API key (live AI)
./deploy.sh YOUR_PROJECT_ID us-central1 YOUR_ANTHROPIC_API_KEY

# Or deploy in mock mode (no API key needed)
./deploy.sh YOUR_PROJECT_ID us-central1
```

### **Option 2: Manual Deployment**

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Deploy to Cloud Run
gcloud run deploy ai-financial-cleaner \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --timeout 300 \
  --set-env-vars "ANTHROPIC_API_KEY=your-key-here,FLASK_ENV=production,ENABLE_AI=true"
```

### **Option 3: Local Testing**

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key"
export FLASK_ENV="development"

# Run locally
python app.py

# Test the local deployment
curl http://localhost:8080/health
curl -X POST http://localhost:8080/demo
```

## 🧪 **Testing Your Deployment**

### **1. Automated Testing**

```bash
# Install requests if not available
pip install requests

# Test your deployment
python test_deployment.py https://your-service-url.run.app
```

### **2. Manual Testing**

```bash
# Get your service URL
SERVICE_URL=$(gcloud run services describe ai-financial-cleaner --region=us-central1 --format='value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test demo endpoint
curl -X POST $SERVICE_URL/demo

# Test custom data processing
curl -X POST $SERVICE_URL/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "merchant": ["PAYPAL*DIGITALOCEAN", "SQ *COFFEE SHOP NYC"],
      "amount": [50.00, 4.50],
      "description": ["Hosting", "Coffee"]
    },
    "config": {
      "enable_ai": true
    }
  }'
```

## 🔧 **Configuration & Environment Variables**

### **Required Environment Variables**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port | 8080 | No |
| `FLASK_ENV` | Flask environment | production | No |
| `ANTHROPIC_API_KEY` | Anthropic API key for live AI | None | No* |
| `ENABLE_AI` | Enable/disable AI processing | true | No |
| `AI_CONFIDENCE_THRESHOLD` | AI confidence threshold | 0.7 | No |
| `MAX_FILE_SIZE` | Max file size in MB | 50 | No |

*Service works in mock mode without API key

### **Updating Environment Variables**

```bash
# Update environment variables on deployed service
gcloud run services update ai-financial-cleaner \
  --region us-central1 \
  --set-env-vars "ANTHROPIC_API_KEY=new-key,ENABLE_AI=true"
```

## 📊 **Monitoring & Maintenance**

### **View Logs**

```bash
# View recent logs
gcloud run services logs read ai-financial-cleaner --region=us-central1

# Follow logs in real-time
gcloud run services logs tail ai-financial-cleaner --region=us-central1
```

### **Monitor Performance**

```bash
# Get service details
gcloud run services describe ai-financial-cleaner --region=us-central1

# View metrics in Cloud Console
# Visit: https://console.cloud.google.com/run
```

### **Update Service**

```bash
# Redeploy with new code
gcloud run deploy ai-financial-cleaner \
  --source . \
  --region us-central1

# Update just the image
gcloud run deploy ai-financial-cleaner \
  --image gcr.io/YOUR_PROJECT/ai-financial-cleaner \
  --region us-central1
```

## 💰 **Cost Management**

### **Expected Costs**

**Cloud Run Costs:**
- **CPU**: ~$0.000024 per vCPU-second
- **Memory**: ~$0.0000025 per GiB-second  
- **Requests**: ~$0.0000004 per request
- **Typical cost**: $10-50/month for moderate usage

**AI Costs (with Anthropic API):**
- **Vendor resolution**: $0.01 per request
- **Category assignment**: $0.01 per request
- **Typical cost**: $0.02 per transaction processed

### **Cost Optimization**

1. **Use AI sparingly**: Enable `use_ai_for_unmatched_only` config
2. **Set memory limits**: Use appropriate memory allocation
3. **Monitor usage**: Check `/stats` endpoint regularly
4. **Cache results**: AI client has built-in caching

## 🔒 **Security Best Practices**

### **API Key Security**

```bash
# Store API key in Secret Manager (recommended)
gcloud secrets create anthropic-api-key --data-file=api-key.txt

# Use secret in Cloud Run
gcloud run deploy ai-financial-cleaner \
  --set-secrets="ANTHROPIC_API_KEY=anthropic-api-key:latest"
```

### **Access Control**

```bash
# Remove public access (if needed)
gcloud run services remove-iam-policy-binding ai-financial-cleaner \
  --region=us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"

# Add specific users
gcloud run services add-iam-policy-binding ai-financial-cleaner \
  --region=us-central1 \
  --member="user:someone@example.com" \
  --role="roles/run.invoker"
```

## 🚨 **Troubleshooting**

### **Common Issues**

**1. Build Failures**
```bash
# Check build logs
gcloud builds list --limit=5

# View specific build
gcloud builds log BUILD_ID
```

**2. Service Not Starting**
```bash
# Check service logs
gcloud run services logs read ai-financial-cleaner --region=us-central1 --limit=50

# Check service configuration
gcloud run services describe ai-financial-cleaner --region=us-central1
```

**3. API Key Issues**
```bash
# Test API key locally
export ANTHROPIC_API_KEY="your-key"
python -c "from llm_client import LLMClient; client = LLMClient(); print(client.resolve_vendor('test'))"
```

**4. Memory Issues**
```bash
# Increase memory allocation
gcloud run deploy ai-financial-cleaner \
  --memory 4Gi \
  --region us-central1
```

### **Health Checks**

```bash
# Check service health
curl https://your-service-url.run.app/health

# Check readiness
curl https://your-service-url.run.app/ready

# Check configuration
curl https://your-service-url.run.app/config
```

## 📞 **Support**

If you encounter issues:

1. **Check logs**: `gcloud run services logs read ai-financial-cleaner`
2. **Verify configuration**: Use `/config` endpoint
3. **Test locally**: Run the service locally first
4. **Check quotas**: Ensure you haven't hit API limits

## 🎉 **Success!**

Once deployed successfully, your AI-Enhanced Financial Cleaner will be available at:

```
https://ai-financial-cleaner-[hash].run.app
```

**Available endpoints:**
- `GET /health` - Service health check
- `GET /stats` - Usage statistics
- `POST /process` - Process financial data
- `POST /demo` - Demo with sample data

**Your service is now ready to process financial data with AI-powered vendor standardization and intelligent categorization!** 🚀 
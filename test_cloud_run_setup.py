#!/usr/bin/env python3
"""
Test script to verify Cloud Run deployment setup
Run this locally to ensure everything is configured correctly before deployment
"""

import os
import sys
import subprocess
import json
import requests
from datetime import datetime

def check_requirements():
    """Check if all requirements are met for deployment"""
    print("🔍 Checking deployment requirements...")
    
    # Check if files exist
    required_files = [
        'app_v5.py',
        'Dockerfile', 
        'requirements.txt',
        'cloudbuild.yaml',
        'deploy-to-cloud-run.sh'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")
    
    # Check if Docker is available
    try:
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found or not working")
        return False
    
    # Check if gcloud is available
    try:
        subprocess.run(['gcloud', '--version'], capture_output=True, check=True)
        print("✅ Google Cloud SDK is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Google Cloud SDK not found")
        return False
    
    return True

def test_local_build():
    """Test if the Docker image builds locally"""
    print("\n🔨 Testing Docker build...")
    
    try:
        result = subprocess.run([
            'docker', 'build', '-t', 'financial-cleaner-test', '.'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Docker build successful")
            return True
        else:
            print(f"❌ Docker build failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out")
        return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False

def test_local_run():
    """Test if the Docker container runs locally"""
    print("\n🚀 Testing local container run...")
    
    try:
        # Start container in background
        container = subprocess.Popen([
            'docker', 'run', '-d', '-p', '8080:8080', 
            '-e', 'ANTHROPIC_API_KEY=test_key',
            'financial-cleaner-test'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        container_id = container.stdout.read().strip()
        
        if not container_id:
            print("❌ Failed to start container")
            return False
        
        print(f"✅ Container started: {container_id[:12]}")
        
        # Wait a bit for startup
        import time
        time.sleep(5)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:8080/health', timeout=10)
            if response.status_code == 200:
                print("✅ Health check passed")
                success = True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                success = False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            success = False
        
        # Clean up
        subprocess.run(['docker', 'stop', container_id], capture_output=True)
        subprocess.run(['docker', 'rm', container_id], capture_output=True)
        
        return success
        
    except Exception as e:
        print(f"❌ Container test error: {e}")
        return False

def check_environment_config():
    """Check environment configuration"""
    print("\n🔧 Checking environment configuration...")
    
    # Check if ANTHROPIC_API_KEY is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print("✅ ANTHROPIC_API_KEY is set")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set (you'll need to set this in Cloud Run)")
    
    # Check gcloud project
    try:
        result = subprocess.run([
            'gcloud', 'config', 'get-value', 'project'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            project = result.stdout.strip()
            print(f"✅ gcloud project set: {project}")
        else:
            print("❌ No gcloud project set")
            return False
    except Exception as e:
        print(f"❌ gcloud config error: {e}")
        return False
    
    return True

def generate_deployment_command():
    """Generate the deployment command"""
    print("\n📋 Deployment commands:")
    
    try:
        result = subprocess.run([
            'gcloud', 'config', 'get-value', 'project'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            project = result.stdout.strip()
            print(f"./deploy-to-cloud-run.sh {project}")
            print("\nOr manual deployment:")
            print(f"gcloud builds submit --tag gcr.io/{project}/financial-cleaner-api")
            print(f"gcloud run deploy financial-cleaner-api --image gcr.io/{project}/financial-cleaner-api --region us-central1 --allow-unauthenticated")
        else:
            print("Set your project first: gcloud config set project YOUR_PROJECT_ID")
    except Exception as e:
        print(f"Error generating commands: {e}")

def main():
    """Main test function"""
    print("🚀 Cloud Run Deployment Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run all checks
    if not check_requirements():
        all_passed = False
    
    if not check_environment_config():
        all_passed = False
    
    if not test_local_build():
        all_passed = False
    
    if not test_local_run():
        all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your setup is ready for Cloud Run deployment!")
        generate_deployment_command()
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before deploying")
        return 1
    
    print("\n💡 Next steps:")
    print("1. Set ANTHROPIC_API_KEY in Cloud Run after deployment")
    print("2. Test the deployed API with the health endpoint")
    print("3. Integrate with your frontend")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
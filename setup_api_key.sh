#!/bin/bash

# Set the Anthropic API key (provide your key via environment before running)
# Example usage:
#   export ANTHROPIC_API_KEY="your-api-key"
#   ./setup_api_key.sh
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "ANTHROPIC_API_KEY is not set. Please export it in your shell before running this script."
  exit 1
fi

# Optionally persist for future sessions (uncomment if desired)
# echo "export ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\"" >> ~/.zshrc

echo "✅ Anthropic API key set successfully!"
echo "🔑 API Key configured for live AI"
echo ""
echo "Now restarting the server with live AI..."

# Kill any existing server
pkill -f "python3 app_v5.py" 2>/dev/null

# Start the server with the API key
python3 app_v5.py
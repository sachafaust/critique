#!/bin/bash
set -e

# Function to handle graceful shutdown
graceful_shutdown() {
    echo "Received shutdown signal, gracefully shutting down..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Function to check health
health_check() {
    python -c "
from llm_critique.core.monitoring import HealthChecker
import json
import sys

try:
    health = HealthChecker.get_system_health()
    api_health = HealthChecker.check_api_connectivity()
    
    if health['status'] == 'healthy':
        print('Health check passed')
        sys.exit(0)
    else:
        print('Health check failed:', health.get('error', 'Unknown error'))
        sys.exit(1)
except Exception as e:
    print('Health check error:', str(e))
    sys.exit(1)
"
}

# Function to start metrics server
start_metrics_server() {
    python -c "
from llm_critique.core.monitoring import MetricsCollector, HealthChecker
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import os

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            health = HealthChecker.get_system_health()
            self.send_response(200 if health['status'] == 'healthy' else 503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health).encode())
        elif self.path == '/metrics':
            metrics = MetricsCollector.get_metrics_summary()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

port = int(os.environ.get('LLM_CRITIQUE_METRICS_PORT', 8080))
server = HTTPServer(('0.0.0.0', port), MetricsHandler)
print(f'Starting metrics server on port {port}')
server.serve_forever()
" &
}

# Initialize environment
echo "Initializing LLM Critique..."

# Set default environment variables if not provided
export LLM_CRITIQUE_LOG_LEVEL=${LLM_CRITIQUE_LOG_LEVEL:-INFO}
export LLM_CRITIQUE_LOG_FORMAT=${LLM_CRITIQUE_LOG_FORMAT:-json}
export LLM_CRITIQUE_MAX_INPUT_TOKENS=${LLM_CRITIQUE_MAX_INPUT_TOKENS:-8000}
export LLM_CRITIQUE_MAX_OUTPUT_TOKENS=${LLM_CRITIQUE_MAX_OUTPUT_TOKENS:-2000}
export LLM_CRITIQUE_MAX_COST_PER_REQUEST=${LLM_CRITIQUE_MAX_COST_PER_REQUEST:-1.0}
export LLM_CRITIQUE_ENABLE_METRICS=${LLM_CRITIQUE_ENABLE_METRICS:-true}

# Check if running in health check mode
if [ "$1" = "health" ]; then
    health_check
    exit $?
fi

# Start metrics server if enabled
if [ "${LLM_CRITIQUE_ENABLE_METRICS}" = "true" ]; then
    echo "Starting metrics server..."
    start_metrics_server
fi

# Validate configuration
echo "Validating configuration..."
python -c "
from llm_critique.config import load_config
try:
    config = load_config()
    print(f'Configuration loaded successfully')
    print(f'Log level: {config.log_level}')
    print(f'Max input tokens: {config.max_input_tokens}')
    print(f'Max cost per request: \${config.max_cost_per_request}')
except Exception as e:
    print(f'Configuration error: {e}')
    exit(1)
"

# Check API keys
echo "Checking API key configuration..."
api_keys_found=0
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OpenAI API key configured"
    api_keys_found=$((api_keys_found + 1))
fi
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ Anthropic API key configured"
    api_keys_found=$((api_keys_found + 1))
fi
if [ -n "$GOOGLE_API_KEY" ]; then
    echo "✓ Google API key configured"
    api_keys_found=$((api_keys_found + 1))
fi
if [ -n "$XAI_API_KEY" ]; then
    echo "✓ X AI API key configured"
    api_keys_found=$((api_keys_found + 1))
fi

if [ $api_keys_found -eq 0 ]; then
    echo "⚠️  Warning: No API keys configured. Limited functionality available."
else
    echo "✓ $api_keys_found API provider(s) configured"
fi

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    echo "Starting LLM Critique CLI..."
    exec llm-critique --help
fi

# Execute the command
echo "Executing: $@"
exec "$@" 
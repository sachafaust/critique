# Production Deployment Guide

This guide covers deploying LLM Critique in production environments with proper security, monitoring, and operational practices.

## 🚀 Quick Start

### Docker Deployment

```bash
# 1. Clone and configure
git clone <repository>
cd llm-critique

# 2. Configure environment
cp docker/config/production.env .env
# Edit .env with your API keys and settings

# 3. Deploy with Docker Compose
docker-compose up -d

# 4. Verify deployment
curl http://localhost:8080/health
```

### Kubernetes Deployment

```bash
# 1. Create namespace
kubectl create namespace llm-critique

# 2. Create secrets
kubectl create secret generic llm-critique-secrets \
  --from-literal=OPENAI_API_KEY=your-key \
  --from-literal=ANTHROPIC_API_KEY=your-key \
  -n llm-critique

# 3. Deploy
kubectl apply -f k8s/ -n llm-critique
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_CRITIQUE_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LLM_CRITIQUE_LOG_FORMAT` | `json` | Log format (json, text) |
| `LLM_CRITIQUE_MAX_INPUT_TOKENS` | `8000` | Maximum input tokens per request |
| `LLM_CRITIQUE_MAX_OUTPUT_TOKENS` | `2000` | Maximum output tokens per request |
| `LLM_CRITIQUE_MAX_COST_PER_REQUEST` | `1.0` | Maximum cost per request (USD) |
| `LLM_CRITIQUE_MAX_FILE_SIZE_MB` | `10` | Maximum file size in MB |
| `LLM_CRITIQUE_API_TIMEOUT` | `120` | API request timeout (seconds) |
| `LLM_CRITIQUE_MAX_RETRIES` | `3` | Maximum API retries |
| `LLM_CRITIQUE_RATE_LIMIT_RPM` | `60` | Rate limit (requests per minute) |
| `LLM_CRITIQUE_ENABLE_METRICS` | `true` | Enable metrics collection |
| `LLM_CRITIQUE_METRICS_PORT` | `8080` | Metrics server port |

### API Keys

Required environment variables for LLM providers:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google
export GOOGLE_API_KEY=AIza...

# X AI
export XAI_API_KEY=xai-...
```

## 🔒 Security

### Container Security

- **Non-root user**: Containers run as `crituser` (UID 1000)
- **Read-only filesystem**: Where possible
- **No new privileges**: Security option enabled
- **Resource limits**: CPU and memory constraints
- **Security scanning**: Trivy scans in CI/CD

### API Key Management

```bash
# Use secrets management
kubectl create secret generic api-keys \
  --from-literal=openai-key=sk-... \
  --from-literal=anthropic-key=sk-ant-...

# Or use external secret managers
# - AWS Secrets Manager
# - Azure Key Vault  
# - HashiCorp Vault
# - Google Secret Manager
```

### Network Security

```yaml
# Network policies (example)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-critique-netpol
spec:
  podSelector:
    matchLabels:
      app: llm-critique
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS to LLM APIs
```

## 📊 Monitoring

### Health Checks

```bash
# System health
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llm-critique'
    static_configs:
      - targets: ['llm-critique:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboards

Key metrics to monitor:

- **Request Rate**: Requests per minute
- **Error Rate**: Failed requests percentage  
- **Response Time**: P50, P90, P99 latencies
- **Cost**: API costs per hour/day
- **Token Usage**: Input/output tokens
- **System Resources**: CPU, memory, disk

### Alerting Rules

```yaml
# Example Prometheus alerts
groups:
- name: llm-critique
  rules:
  - alert: HighErrorRate
    expr: rate(llm_critique_errors_total[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighCost
    expr: rate(llm_critique_cost_total[1h]) > 10
    for: 5m
    annotations:
      summary: "High API costs detected"
      
  - alert: ServiceDown
    expr: up{job="llm-critique"} == 0
    for: 1m
    annotations:
      summary: "LLM Critique service is down"
```

## 🔄 Deployment Strategies

### Blue-Green Deployment

```bash
# Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# Test green environment
curl http://green.llm-critique.local/health

# Switch traffic (update load balancer)
# Terminate blue environment
docker-compose -f docker-compose.blue.yml down
```

### Rolling Updates (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-critique
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: llm-critique
        image: ghcr.io/your-org/llm-critique:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## 🚨 Incident Response

### Common Issues

#### High API Costs

```bash
# Check cost metrics
curl http://localhost:8080/metrics | grep cost

# Adjust limits
export LLM_CRITIQUE_MAX_COST_PER_REQUEST=0.5

# Restart service
docker-compose restart llm-critique
```

#### Rate Limiting

```bash
# Check rate limit metrics
curl http://localhost:8080/metrics | grep rate_limit

# Adjust rate limits
export LLM_CRITIQUE_RATE_LIMIT_RPM=30

# Or implement backoff in client
```

#### Memory Issues

```bash
# Check memory usage
docker stats llm-critique

# Increase memory limits
# Update docker-compose.yml or k8s deployment
```

### Debugging

```bash
# Enable debug logging
export LLM_CRITIQUE_LOG_LEVEL=DEBUG

# View logs
docker logs -f llm-critique

# Or in Kubernetes
kubectl logs -f deployment/llm-critique -n llm-critique
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-critique-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-critique
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing

```nginx
# nginx.conf
upstream llm_critique {
    least_conn;
    server llm-critique-1:8080;
    server llm-critique-2:8080;
    server llm-critique-3:8080;
}

server {
    listen 80;
    location / {
        proxy_pass http://llm_critique;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 120s;
    }
    
    location /health {
        proxy_pass http://llm_critique;
        access_log off;
    }
}
```

## 🔧 Maintenance

### Backup

```bash
# Backup configuration
tar -czf llm-critique-config-$(date +%Y%m%d).tar.gz \
  .env config/ docker-compose.yml

# Backup logs (if persistent)
tar -czf llm-critique-logs-$(date +%Y%m%d).tar.gz logs/
```

### Updates

```bash
# Pull latest image
docker pull ghcr.io/your-org/llm-critique:latest

# Update with zero downtime
docker-compose up -d --no-deps llm-critique

# Verify update
curl http://localhost:8080/health
```

### Log Rotation

```yaml
# docker-compose.yml
services:
  llm-critique:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## 🧪 Testing

### Health Check Testing

```bash
#!/bin/bash
# health-check.sh

ENDPOINT="http://localhost:8080"

# Test health endpoint
if curl -f "$ENDPOINT/health" > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test metrics endpoint
if curl -f "$ENDPOINT/metrics" > /dev/null 2>&1; then
    echo "✅ Metrics endpoint accessible"
else
    echo "❌ Metrics endpoint failed"
    exit 1
fi
```

### Load Testing

```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:8080/health

# Using wrk
wrk -t12 -c400 -d30s http://localhost:8080/health
```

## 📋 Checklist

### Pre-deployment

- [ ] API keys configured
- [ ] Resource limits set
- [ ] Security scanning passed
- [ ] Health checks working
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Post-deployment

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being generated
- [ ] Alerts configured
- [ ] Performance baseline established
- [ ] Documentation updated

## 🆘 Support

### Logs Analysis

```bash
# Find errors in logs
docker logs llm-critique 2>&1 | grep -i error

# Monitor real-time logs
docker logs -f llm-critique | jq '.'

# Search for specific patterns
docker logs llm-critique 2>&1 | grep "circuit_breaker"
```

### Performance Tuning

```bash
# Adjust worker processes
export LLM_CRITIQUE_WORKERS=4

# Tune memory settings
export LLM_CRITIQUE_MAX_MEMORY=2048

# Optimize for your workload
export LLM_CRITIQUE_MAX_INPUT_TOKENS=4000  # Smaller for faster processing
```

---

For additional support, check the [troubleshooting guide](TROUBLESHOOTING.md) or open an issue on GitHub. 
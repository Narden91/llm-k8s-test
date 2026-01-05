# CLI Reference Guide

Quick reference for common commands used with the LLM Chat Platform.

---

## 🐍 Python Environment (uv)

```bash
# Initialize environment and install dependencies
uv sync

# Run Streamlit chat application
uv run streamlit run streamlit_app/app.py

# Add/remove packages
uv add <package>
uv remove <package>

# Update lock file
uv lock --upgrade
```

---

## 🐳 Docker

### LLM Chat Application
```bash
# Build container image
docker build -f Dockerfile.llm -t llm-chat .

# Run locally (requires NVIDIA GPU)
docker run --gpus all -p 8501:8501 llm-chat

# Run with Hugging Face token (for gated models)
docker run --gpus all -p 8501:8501 \
  -e HF_TOKEN=your_token \
  llm-chat

# Interactive shell for debugging
docker run --gpus all -it --entrypoint /bin/bash llm-chat
```

---

## ☸️ Kubernetes

### Secrets Setup
```bash
# GitHub Container Registry
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PAT \
  --docker-email=YOUR_EMAIL

# Hugging Face token (optional, for gated models)
kubectl create secret generic llm-secrets \
  --from-literal=HF_TOKEN=your_token
```

### Deploy LLM Chat
```bash
# Deploy application
kubectl apply -f llm-manifest.yml

# Check deployment status
kubectl get pods -l app=llm-chat

# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=llm-chat --timeout=600s
```

### Port Forwarding
```bash
# Access Streamlit UI locally
kubectl port-forward svc/llm-chat-service 8501:8501

# Then open: http://localhost:8501
```

### Monitoring
```bash
# Pod status
kubectl get pods -l app=llm-chat
kubectl describe pod -l app=llm-chat

# View logs (follow mode)
kubectl logs -f -l app=llm-chat

# Check GPU status inside container
kubectl exec -it $(kubectl get pods -l app=llm-chat -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi

# Check GPU memory usage
kubectl exec -it $(kubectl get pods -l app=llm-chat -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Cleanup
```bash
# Delete all LLM resources
kubectl delete -f llm-manifest.yml

# Delete secrets
kubectl delete secret ghcr-secret llm-secrets
```

---

## 🏷️ Git Tagging (CI/CD)

```bash
# Trigger LLM container build
git tag llm-v1.0.0
git push origin llm-v1.0.0

# List existing tags
git tag -l "llm-*"

# Delete a tag (local + remote)
git tag -d llm-v1.0.0
git push origin --delete llm-v1.0.0
```

---

## 🚀 Streamlit (Local Development)

```bash
# Run LLM chat locally (requires GPU + vLLM)
uv run streamlit run streamlit_app/app.py

# With custom port
uv run streamlit run streamlit_app/app.py --server.port 8502

# Headless mode (for servers)
uv run streamlit run streamlit_app/app.py \
  --server.headless true \
  --server.address 0.0.0.0
```

---

## 🧪 Verification

```bash
# Verify S3 connection
python verify_s3.py

# Check Python imports (will fail without vLLM/GPU)
python -c "from llm_operations import LLMEngine; print('OK')"

# Test Streamlit app syntax
python -m py_compile streamlit_app/app.py
```

---

## 📊 Useful Aliases

Add these to your shell profile for convenience:

```bash
# Bash/Zsh aliases
alias llm-logs='kubectl logs -f -l app=llm-chat'
alias llm-status='kubectl get pods -l app=llm-chat'
alias llm-forward='kubectl port-forward svc/llm-chat-service 8501:8501'
alias llm-gpu='kubectl exec -it $(kubectl get pods -l app=llm-chat -o jsonpath="{.items[0].metadata.name}") -- nvidia-smi'

# PowerShell functions
function llm-logs { kubectl logs -f -l app=llm-chat }
function llm-status { kubectl get pods -l app=llm-chat }
function llm-forward { kubectl port-forward svc/llm-chat-service 8501:8501 }
```

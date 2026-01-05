# CLI Reference Guide

Quick reference for common commands used in this repository.

---

## 🐍 Python Environment (uv)

```bash
# Initialize environment and install deps
uv sync

# Run application
uv run python main.py

# Add/remove packages
uv add <package>
uv remove <package>

# Update lock file
uv lock --upgrade
```

---

## 🐳 Docker

### Benchmark Application
```bash
# Build
docker build -t k8s-test .

# Run locally
docker run --gpus all k8s-test
```

### LLM Chat Application
```bash
# Build
docker build -f Dockerfile.llm -t llm-chat .

# Run locally (requires NVIDIA GPU)
docker run --gpus all -p 8501:8501 llm-chat
```

---

## ☸️ Kubernetes

### Secrets Setup
```bash
# GitHub Container Registry
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PAT

# S3 credentials
kubectl create secret generic s3-secrets \
  --from-literal=S3_ENDPOINT_URL=your-endpoint \
  --from-literal=AWS_ACCESS_KEY_ID=your-key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your-secret \
  --from-literal=S3_BUCKET=your-bucket

# Hugging Face token (for LLM)
kubectl create secret generic llm-secrets \
  --from-literal=HF_TOKEN=your_token
```

### Deploy Applications
```bash
# Benchmark app
kubectl apply -f prj-configmap.yml
kubectl apply -f manifest.yml

# LLM chat app
kubectl apply -f llm-manifest.yml
```

### Monitoring
```bash
# Pod status
kubectl get pods
kubectl describe pod <pod-name>

# Logs (follow mode)
kubectl logs -f <pod-name>

# GPU status
kubectl exec -it <pod-name> -- nvidia-smi

# Port forward (access Streamlit)
kubectl port-forward svc/llm-chat-service 8501:8501
```

### Cleanup
```bash
# Delete pods
kubectl delete pod k8s-test
kubectl delete pod llm-chat

# Delete all resources from manifest
kubectl delete -f llm-manifest.yml
```

---

## 🏷️ Git Tagging (CI/CD)

```bash
# Trigger benchmark container build
git tag v1.0.0
git push origin v1.0.0

# Trigger LLM container build
git tag llm-v1.0.0
git push origin llm-v1.0.0
```

---

## 🚀 Streamlit (Local Dev)

```bash
# Run LLM chat locally
streamlit run streamlit_app/app.py

# With specific port
streamlit run streamlit_app/app.py --server.port 8502
```

---

## 🧪 Testing

```bash
# Verify S3 connection
python verify_s3.py

# Run with specific mode
python main.py --mode matrix
python main.py --mode image

# Custom matrix sizes
python main.py --mode matrix --matrix-sizes 1000,2000,5000
```

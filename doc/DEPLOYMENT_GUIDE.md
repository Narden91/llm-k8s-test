# LLM Chat Deployment Guide

Quick guide to build, deploy, and access the LLM Chat application on Kubernetes.

---

## Workflow Overview

```
Local Changes → Git Push → GitHub Actions (build) → GHCR → K8s Deploy → Access via NodePort
```

1. **Edit code locally**
2. **Push + tag** triggers GitHub Actions
3. **Docker image** built and pushed to GHCR
4. **Apply manifest** pulls image to K8s
5. **Access app** via NodeIP:30851

---

## Step-by-Step Deployment

### 1. Stage and Commit Changes

```bash
# Add all changes
git add .

# Commit with descriptive message
git commit -m "feat: your feature description"

# Push to main
git push origin main
```

### 2. Trigger Docker Build

```bash
# Create version tag (triggers GitHub Actions)
git tag llm-v1.2.0

# Push tag to trigger build
git push origin llm-v1.2.0
```

> **Wait** for GitHub Actions to complete (~5-10 min). Check status at:  
> `https://github.com/<your-repo>/actions`

### 3. Create Secrets (First Time Only)

```bash
# GitHub Container Registry access
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_GITHUB_USER \
  --docker-password=YOUR_GITHUB_PAT

# HuggingFace token (for gated models)
kubectl create secret generic llm-secrets \
  --from-literal=HF_TOKEN=your_hf_token

# WandB API key (for monitoring)
kubectl create secret generic wandb-secrets \
  --from-literal=WANDB_API_KEY=your_wandb_key
```

### 4. Deploy to Kubernetes

```bash
# Delete existing pod (if updating)
kubectl delete pod llm-chat --ignore-not-found

# Apply manifest (creates Pod, Service, ConfigMap)
kubectl apply -f llm-manifest.yml

# Watch pod status
kubectl get pods -w
```

### 5. Monitor Startup

```bash
# View logs (model download takes a few minutes)
kubectl logs -f llm-chat

# Check GPU status
kubectl exec -it llm-chat -- nvidia-smi
```

---

## Accessing the Application

### Get Node IP

```bash
# List nodes with IPs
kubectl get nodes -o wide

# Example output:
# NAME       STATUS   INTERNAL-IP    EXTERNAL-IP
# node-1     Ready    192.168.1.10   <none>
```

### Access URL

```
http://<NODE-IP>:30851
```

**Example**: If node IP is `192.168.1.10`:
```
http://192.168.1.10:30851
```

### Alternative: Port Forward (if NodePort blocked)

```bash
kubectl port-forward svc/llm-chat-service 8501:8501
# Then access: http://localhost:8501
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Commit | `git add . && git commit -m "msg"` |
| Push | `git push origin main` |
| Tag + Build | `git tag llm-vX.Y.Z && git push origin llm-vX.Y.Z` |
| Deploy | `kubectl apply -f llm-manifest.yml` |
| Logs | `kubectl logs -f llm-chat` |
| GPU Status | `kubectl exec -it llm-chat -- nvidia-smi` |
| Get Node IP | `kubectl get nodes -o wide` |
| Delete Pod | `kubectl delete pod llm-chat` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ImagePullBackOff | Check `ghcr-secret` is correct |
| OOMKilled | Reduce GPU memory in sidebar |
| Model not loading | Check HF token for gated models |
| Can't access NodePort | Use port-forward instead |

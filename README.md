# 🤖 LLM Chat Platform on Kubernetes

## 👤 Author
**Emanuele Nardone**

## 📌 Overview
A **production-ready LLM inference platform** powered by **vLLM** and **Mistral 7B**, designed for deployment on **GPU-enabled Kubernetes clusters** (NVIDIA A6000). Features a modern **Streamlit chat interface** with streaming responses.

---

## ✨ Features

### 🧠 LLM Inference Engine
- **vLLM** for high-throughput, low-latency inference
- **Mistral 7B Instruct** with proper prompt formatting
- Configurable generation parameters (temperature, top-p, max tokens)
- Multi-turn conversation history management
- Streaming token generation for real-time responses

### 💬 Chat Interface
- Modern **Streamlit** web UI
- Real-time streaming responses with typing indicator
- Configurable model settings via sidebar
- Session-based conversation management
- Mobile-responsive design

### 🏗️ Infrastructure
- **NVIDIA CUDA** runtime optimized containers
- **Kubernetes** deployment with GPU scheduling
- Health checks and readiness probes
- GitHub Actions CI/CD pipeline
- Hugging Face model caching

---

## 📁 Project Structure
```bash
llm-k8s-test/
├── config/                   # 📜 Configuration handlers
│   └── s3_config_handler.py
├── configs/                  # ⚙️ YAML configuration files
│   └── llm_config.yaml
├── llm_operations/           # 🧠 LLM inference engine
│   ├── llm_config.py         # Pydantic configuration models
│   └── llm_inference.py      # vLLM engine wrapper
├── s3_operations/            # ☁️ S3 storage utilities
│   ├── s3_client.py          # Low-level S3 client wrapper
│   └── s3_operations.py      # High-level S3 operations
├── streamlit_app/            # 💬 Chat interface
│   └── app.py                # Streamlit application
├── doc/                      # 📚 Documentation
├── .github/workflows/        # 🔄 CI/CD pipelines
├── Dockerfile.llm            # 🐳 LLM container image
├── llm-manifest.yml          # ☸️ Kubernetes deployment
├── requirements-llm.txt      # 📦 Python dependencies
├── verify_s3.py              # 🔍 S3 Setup Verification Script
└── pyproject.toml            # 🔧 Project configuration
```

---

## ⚠️ Prerequisites
- **Kubernetes cluster** with NVIDIA GPU support
- **NVIDIA Container Toolkit** installed
- **NVIDIA A6000** (or compatible GPU with ≥16GB VRAM)
- **Hugging Face token** (optional, for gated models)
- **kubectl** CLI tool

---

## 🚀 Quick Start

### Local Development

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run Streamlit app (requires GPU)
uv run streamlit run streamlit_app/app.py

# Verify S3 connection (optional)
uv run verify_s3.py
```

### Kubernetes Deployment

```bash
# Create GitHub Container Registry secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_GITHUB_USERNAME \
  --docker-password=YOUR_GITHUB_PAT \
  --docker-email=YOUR_GITHUB_EMAIL

# Create Hugging Face secret (optional, for gated models)
kubectl create secret generic llm-secrets \
  --from-literal=HF_TOKEN=your-hf-token

# Deploy the application
kubectl apply -f llm-manifest.yml

# Check pod status
kubectl get pods -l app=llm-chat

# View logs
kubectl logs -f -l app=llm-chat

# Port forward to access UI
kubectl port-forward svc/llm-chat-service 8501:8501
```

Then open http://localhost:8501 in your browser.

---

## 🔧 Configuration

### Generation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Sampling temperature (0-2). Higher = more creative |
| `max_tokens` | 2048 | Maximum response length |
| `top_p` | 0.95 | Nucleus sampling threshold |
| `top_k` | 50 | Top-k sampling |
| `repetition_penalty` | 1.1 | Token repetition penalty |

### Model Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `mistralai/Mistral-7B-Instruct-v0.3` | Hugging Face model |
| `gpu_memory_utilization` | 0.9 | GPU memory fraction (0.1-0.99) |
| `tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |

---

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows for automated container builds:

- **Triggers** on version tags (`v*.*.*`)
- **Builds** optimized vLLM-based container image
- **Pushes** to GitHub Container Registry (GHCR)
- **Tags** with semantic version and Git SHA

### Triggering a Build
```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## 🔍 Monitoring & Troubleshooting

### Check Pod Status
```bash
kubectl get pods -l app=llm-chat
kubectl describe pod -l app=llm-chat
```

### View Logs
```bash
kubectl logs -f -l app=llm-chat
```

### Check GPU Status
```bash
kubectl exec -it $(kubectl get pods -l app=llm-chat -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Model loading timeout | Increase `initialDelaySeconds` in readiness probe |
| OOM errors | Reduce `gpu_memory_utilization` or use smaller model |
| Slow first response | Normal - KV cache warmup on first request |

---

## 📚 Documentation

Additional documentation available in the `doc/` folder:
- [CLI Reference](doc/CLI_REFERENCE.md)
- [Kubernetes Commands](doc/K8s_commands.md)
- [Seeweb Setup Guide](doc/Seeweb_setup_guide.md)

### Future Project Ideas
- [Multi-Model GPU Orchestration](doc/IDEA_MULTI_MODEL_ORCHESTRATION.md)
- [Federated LLM with Privacy](doc/IDEA_FEDERATED_LLM_PRIVACY.md)
- [Self-Optimizing LLM Inference](doc/IDEA_SELF_OPTIMIZING_LLM.md)

---

## 📜 License
**Unicas & Seeweb**

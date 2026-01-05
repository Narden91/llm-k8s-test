# 🚀 GPU-Accelerated Processing Framework

## 👤 Author
**Emanuele Nardone**

## 📌 Overview
A **containerized application** that performs **GPU-accelerated processing tasks** using PyTorch, designed to run in a **Seeweb Serverless GPU Cluster** with **NVIDIA A6000 GPUs**. The framework supports two primary processing modes:

1. 🔢 **Matrix Multiplication Benchmark**: Compares CPU and GPU performance for various matrix sizes.
2. 🖼 **Image Processing Pipeline**: Applies transformations using both CPU and GPU, including **Gaussian blur** and **color adjustments**.

📂 Results are stored in **S3-compatible storage**, and container images are managed through **GitHub Container Registry (GHCR)**.

---

## ✨ Features

### 🔥 Core Features
✅ Dual processing modes: **Matrix multiplication** and **Image processing**  
✅ **Automatic GPU detection** with CPU fallback  
✅ **Detailed performance metrics** and speedup calculations  
✅ **Rich CLI output** with colored performance indicators  
✅ **S3-compatible storage integration**  
✅ **Comprehensive logging system**  
✅ Configurable via **CLI arguments** or **environment variables**  

### 🔢 Matrix Processing Features
🔹 Configurable matrix sizes for benchmarking  
🔹 PyTorch-based **CPU vs GPU computations**  
🔹 **Memory-efficient** large matrix handling  

### 🖼 Image Processing Features
🎨 Batch image processing capabilities  
🎨 **Gaussian blur** and **color adjustment** transformations  
🎨 Support for **PNG, JPG, JPEG** formats  
🎨 **Parallel processing** optimization  

### 🏗 Infrastructure Features
🛠 **NVIDIA CUDA 11.8.0 runtime** support  
🛠 **Kubernetes integration** with GPU resource management  
🛠 **GitHub Actions CI/CD pipeline**  
🛠 **Exponential backoff retry mechanism** for S3 operations  

---

## 📁 Project Structure
```bash
k8s_test/
├── benchmark_operations/      # 🔢 Matrix multiplication operations
│   └── benchmark_operations.py
├── image_processing/         # 🖼 Image processing operations
│   └── image_processing_operations.py
├── cli_operations/          # 🎛 Command-line interface handling
│   └── cli_operations.py
├── config/                  # ⚙️ Configuration management
│   └── s3_config_handler.py
├── s3_operations/          # ☁️ S3 storage operations
│   ├── s3_client.py       # 🔗 Low-level S3 client with retry logic
│   └── s3_operations.py   # 📦 High-level S3 operations
├── .env.test              # 🌎 Environment variables template
├── .gitignore             # 🚫 Git ignore rules
├── Dockerfile             # 🐳 NVIDIA CUDA-based container configuration
├── main.py               # 🎯 Application entry point
├── python_image.yml      # 📜 Kubernetes pod configuration
└── requirements.txt      # 📌 Python dependencies
```

---

## ⚠️ Prerequisites
🔹 **Kubernetes cluster** with NVIDIA GPU support  
🔹 **NVIDIA Container Toolkit**  
🔹 **Access to GitHub Container Registry**  
🔹 **S3-compatible storage**  
🔹 **kubectl CLI tool**  
🔹 **Python 3.8+** (for local development)  

---

## 📦 Local Development with uv

This project uses [uv](https://github.com/astral-sh/uv) for extremely fast Python package management.

### 📥 Installation
Using the standalone installer (recommended):

```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 🚀 Getting Started

1. **Initialize the Environment**:
   This will create the virtual environment and install all dependencies defined in `pyproject.toml`.
   ```bash
   uv sync
   ```

2. **Run the Application**:
   Execute the script within the managed environment:
   ```bash
   uv run python main.py
   ```

3. **Managing Dependencies**:
   - **Add a new package**:
     ```bash
     uv add <package_name>
     ```
   - **Remove a package**:
     ```bash
     uv remove <package_name>
     ```
   - **Update dependencies**:
     ```bash
     uv lock --upgrade
     ```

---

## 🔧 Environment Variables
Create a `.env` file based on `.env.test`:
```bash
# ☁️ S3 Configuration
S3_ENDPOINT_URL=your-s3-endpoint
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=your-bucket-name

# 🔧 Processing Configuration
PROCESSING_MODE=matrix|image  # Optional, defaults to matrix
MATRIX_SIZES=1000,2000,3000  # Optional for matrix mode
RAW_IMAGES_FOLDER=RawImages  # Optional for image mode
PROCESSED_IMAGES_FOLDER=ProcessedImages  # Optional for image mode
RESULTS_FOLDER=benchmark_results  # Optional
```

---

## 🚀 Kubernetes Deployment

### 🖥 1. GPU Runtime Configuration
```yaml
spec:
  runtimeClassName: seeweb-nvidia-1xa6000
  containers:
    - resources:
        limits:
          nvidia.com/gpu: "1"
```

### 🔑 2. Create Required Secrets
```bash
# GitHub Container Registry credentials
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=YOUR_GITHUB_USERNAME \
  --docker-password=YOUR_GITHUB_PAT \
  --docker-email=YOUR_GITHUB_EMAIL

# S3 credentials
kubectl create secret generic s3-secrets \
  --from-literal=S3_ENDPOINT_URL=your-s3-endpoint \
  --from-literal=AWS_ACCESS_KEY_ID=your-access-key \
  --from-literal=AWS_SECRET_ACCESS_KEY=your-secret-key \
  --from-literal=S3_BUCKET=your-bucket-name
```

### 🛠 3. Create ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prj-configmap
data:
  MATRIX_SIZES: "4000,5000,10000"
  PROCESSING_MODE: "matrix"  # or "image"
  RAW_IMAGES_FOLDER: "RawImages"
  PROCESSED_IMAGES_FOLDER: "ProcessedImages"
  RESULTS_FOLDER: "benchmark_results"
```
Apply with:
```bash
kubectl apply -f prj-configmap.yml
```

### 🚀 4. Launch the Application
```bash
kubectl apply -f manifest.yml
```

---

## 🔄 CI/CD Pipeline
✅ **Triggers on version tags (v*.*.*)**  
✅ **Uses NVIDIA CUDA 11.8.0 base image**  
✅ **Pushes to GitHub Container Registry**  
✅ **Tags images with semantic version and Git SHA**  

### 🚀 Triggering a Build
```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## 🔍 Monitoring and Troubleshooting

### 📌 Pod Status and Logs
```bash
# Check pod status
kubectl get pods
kubectl describe pod k8s-test

# View logs
kubectl logs -f k8s-test

# Check GPU status
kubectl exec -it k8s-test -- nvidia-smi
```

### ⚠️ Common Issues and Solutions
1. **GPU Not Detected**  
   🔹 Verify runtime class configuration  
   🔹 Check NVIDIA device plugin status  
   ```bash
   kubectl get pods -n kube-system | grep nvidia-device-plugin
   ```
2. **S3 Connection Issues**  
   🔹 Verify endpoint and credentials  
   🔹 Check network connectivity  
   🔹 Review exponential backoff settings  
3. **Performance Optimization**  
   🔹 Monitor GPU memory usage  
   🔹 Adjust batch sizes for image processing  
   🔹 Consider matrix size limitations  

---

## 🏁 License
📜 **Unicas & Seeweb**


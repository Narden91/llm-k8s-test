"""Streamlit chat application for LLM interaction with WandB monitoring."""

import os
import sys
import time

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_operations.llm_config import LLMConfig
from llm_operations.llm_inference import ConversationHistory, LLMEngine
from llm_operations.metrics import MetricsLogger, InferenceMetrics

# 2025 Model Presets - Optimized for NVIDIA A6000 (48GB VRAM)
MODEL_PRESETS = {
    # Compact Models (≤20 GB VRAM in 4-bit)
    "⚡ Phi-4 Medium (14B)": {
        "id": "microsoft/Phi-4",
        "vram": "~7-10 GB",
        "best_for": "Reasoning, fast inference",
    },
    "⚡ Gemma 3 27B": {
        "id": "google/gemma-3-27b-it",
        "vram": "~20 GB",
        "best_for": "General, 128K context",
    },
    "⚡ Qwen3 30B MoE": {
        "id": "Qwen/Qwen3-30B-A3B",
        "vram": "~15-20 GB",
        "best_for": "Multilingual, tools",
    },
    "⚡ DeepSeek V3 16B": {
        "id": "deepseek-ai/DeepSeek-V3-Base",
        "vram": "~8-12 GB",
        "best_for": "General, code",
    },
    "🧑‍💻 DeepSeek Coder V2": {
        "id": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "vram": "~16 GB",
        "best_for": "Code generation, math",
    },
    # Large Models (35-45 GB VRAM in 4-bit)
    "🦙 Llama 4 Scout 70B": {
        "id": "meta-llama/Llama-4-Scout-70B",
        "vram": "~35 GB",
        "best_for": "Advanced chat, agents",
    },
    "🔮 Qwen3 72B": {
        "id": "Qwen/Qwen3-72B-Instruct",
        "vram": "~36 GB",
        "best_for": "Research, 128K context",
    },
    # Legacy Models (known working)
    "✅ Mistral 7B v0.2": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "vram": "~14 GB",
        "best_for": "Stable, tested",
    },
    "🚀 TinyLlama 1.1B": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "vram": "~2 GB",
        "best_for": "Ultra-fast testing",
    },
}

# Page configuration
st.set_page_config(
    page_title="LLM Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .main-header { text-align: center; padding: 1rem 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 1rem; }
    .status-indicator { padding: 0.5rem; border-radius: 0.25rem; text-align: center; margin-bottom: 1rem; }
    .status-loading { background-color: #fff3cd; color: #856404; }
    .status-ready { background-color: #d4edda; color: #155724; }
    .model-info { font-size: 0.85rem; color: #666; margin-top: 0.25rem; }
    .wandb-link { background-color: #FFBE00; color: black; padding: 0.5rem; border-radius: 0.25rem; text-align: center; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationHistory(
            system_prompt="You are a helpful, harmless, and honest AI assistant."
        )
    if "llm_engine" not in st.session_state:
        st.session_state.llm_engine = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "metrics_logger" not in st.session_state:
        st.session_state.metrics_logger = None
    if "wandb_url" not in st.session_state:
        st.session_state.wandb_url = None


def start_new_chat() -> None:
    """Start a new chat session, keeping the model loaded."""
    st.session_state.messages = []
    st.session_state.conversation = ConversationHistory(
        system_prompt="You are a helpful, harmless, and honest AI assistant."
    )


def render_sidebar() -> dict:
    """Render the sidebar with model settings."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model Selection
        st.subheader("🤖 Model Selection")
        model_names = list(MODEL_PRESETS.keys())
        
        # Default to Mistral 7B (known working)
        default_idx = model_names.index("✅ Mistral 7B v0.2") if "✅ Mistral 7B v0.2" in model_names else 0
        
        model_name = st.selectbox(
            "Select Model",
            options=model_names,
            index=default_idx,
            disabled=st.session_state.model_loaded,
        )
        
        model_info = MODEL_PRESETS[model_name]
        st.markdown(
            f'<div class="model-info">📦 <code>{model_info["id"]}</code><br>'
            f'💾 VRAM: {model_info["vram"]}<br>'
            f'✨ Best for: {model_info["best_for"]}</div>',
            unsafe_allow_html=True,
        )

        # GPU Settings
        st.divider()
        st.subheader("🎛️ GPU Settings")
        
        gpu_memory = st.slider(
            "GPU Memory Utilization",
            min_value=0.5,
            max_value=0.9,
            value=0.75,
            step=0.05,
            disabled=st.session_state.model_loaded,
        )
        
        context_length = st.select_slider(
            "Max Context Length",
            options=[1024, 2048, 4096, 8192],
            value=2048,
            disabled=st.session_state.model_loaded,
        )

        # Generation Parameters
        st.divider()
        st.subheader("📝 Generation")
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 128, 2048, 1024, 128)
        top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)

        # WandB Settings
        st.divider()
        st.subheader("📊 Monitoring")
        enable_wandb = st.checkbox("Enable WandB Logging", value=True)

        # Actions
        st.divider()
        st.subheader("🎬 Actions")
        
        load_clicked = False
        if st.session_state.model_loaded:
            if st.button("🔄 Unload Model", use_container_width=True):
                if st.session_state.llm_engine:
                    st.session_state.llm_engine.unload_model()
                if st.session_state.metrics_logger:
                    st.session_state.metrics_logger.end_session()
                st.session_state.llm_engine = None
                st.session_state.model_loaded = False
                st.session_state.current_model = None
                st.session_state.metrics_logger = None
                st.session_state.wandb_url = None
                st.session_state.messages = []
                st.rerun()
            
            if st.button("💬 New Chat", use_container_width=True):
                start_new_chat()
                st.rerun()
        else:
            load_clicked = st.button("🚀 Load Model", use_container_width=True, type="primary")

        return {
            "model_id": model_info["id"],
            "model_name": model_name,
            "gpu_memory": gpu_memory,
            "context_length": context_length,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "enable_wandb": enable_wandb,
            "load_clicked": load_clicked,
        }


def load_model(settings: dict) -> None:
    """Load the LLM model with optional WandB logging."""
    model_name = settings["model_name"]
    model_id = settings["model_id"]
    
    with st.spinner(f"Loading {model_name}... This may take a few minutes."):
        try:
            start_time = time.time()
            
            # Initialize metrics logger
            if settings["enable_wandb"]:
                metrics = MetricsLogger(project="llm-chat")
                wandb_url = metrics.start_session(
                    model_name=model_name,
                    model_id=model_id,
                    gpu_memory_utilization=settings["gpu_memory"],
                    context_length=settings["context_length"],
                )
                st.session_state.metrics_logger = metrics
                st.session_state.wandb_url = wandb_url
            
            # Load model
            config = LLMConfig(
                model={
                    "model_id": model_id,
                    "gpu_memory_utilization": settings["gpu_memory"],
                    "max_model_len": settings["context_length"],
                    "trust_remote_code": True,
                }
            )
            engine = LLMEngine(config)
            engine.load_model()
            
            load_time = time.time() - start_time
            
            # Log load time
            if st.session_state.metrics_logger:
                st.session_state.metrics_logger.log_model_load(load_time, model_name)
            
            st.session_state.llm_engine = engine
            st.session_state.model_loaded = True
            st.session_state.current_model = model_name
            
            st.success(f"✅ {model_name} loaded in {load_time:.1f}s!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            if st.session_state.metrics_logger:
                st.session_state.metrics_logger.end_session()


def render_chat_messages() -> None:
    """Render the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_response(settings: dict, user_input: str) -> None:
    """Generate and stream the assistant's response with metrics."""
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        start_time = time.time()
        token_count = 0

        try:
            for token in st.session_state.llm_engine.generate_stream(
                prompt=user_input,
                conversation=st.session_state.conversation,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                top_p=settings["top_p"],
            ):
                full_response += token
                token_count += 1
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            
            # Calculate and log metrics
            latency = time.time() - start_time
            
            if st.session_state.metrics_logger:
                metrics = InferenceMetrics(
                    input_tokens=len(user_input.split()),  # Approximate
                    output_tokens=token_count,
                    latency_seconds=latency,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                )
                metrics.calculate_tokens_per_second()
                st.session_state.metrics_logger.log_inference(metrics)
                st.session_state.metrics_logger.log_gpu_snapshot()
            
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {error_msg}"}
            )


def main() -> None:
    """Main application entry point."""
    initialize_session_state()

    # Header
    model_info = f" • {st.session_state.current_model}" if st.session_state.current_model else ""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🤖 LLM Chat Assistant</h1>
            <p>Powered by vLLM on NVIDIA A6000{model_info}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    settings = render_sidebar()

    # Load model if requested
    if settings["load_clicked"]:
        load_model(settings)

    # WandB link
    if st.session_state.wandb_url:
        st.markdown(
            f'<div class="wandb-link">📊 <a href="{st.session_state.wandb_url}" target="_blank">View WandB Dashboard</a></div>',
            unsafe_allow_html=True,
        )

    # Model status
    if st.session_state.model_loaded:
        st.markdown(
            f'<div class="status-indicator status-ready">✅ {st.session_state.current_model} Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-indicator status-loading">⏳ Select a model and click "Load Model"</div>',
            unsafe_allow_html=True,
        )

    # Chat
    render_chat_messages()

    if prompt := st.chat_input("Type your message...", disabled=not st.session_state.model_loaded):
        generate_response(settings, prompt)


if __name__ == "__main__":
    main()

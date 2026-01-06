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

# Model Presets - Compatible with vLLM 0.4.2 on NVIDIA A6000 (48GB VRAM)
MODEL_PRESETS = {
    "Mistral 7B Instruct v0.2": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "vram": "~14 GB",
        "desc": "General chat, stable baseline",
    },
    "Phi-3 Medium 14B": {
        "id": "microsoft/Phi-3-medium-4k-instruct",
        "vram": "~28 GB",
        "desc": "Reasoning, fast inference",
    },
    "Qwen2 72B Instruct": {
        "id": "Qwen/Qwen2-72B-Instruct",
        "vram": "~40 GB",
        "desc": "Advanced reasoning, multilingual",
    },
    "Llama 3.1 70B Instruct": {
        "id": "meta-llama/Llama-3.1-70B-Instruct",
        "vram": "~40 GB",
        "desc": "Advanced chat, coding",
    },
    "CodeLlama 34B Instruct": {
        "id": "codellama/CodeLlama-34b-Instruct-hf",
        "vram": "~20 GB",
        "desc": "Code generation",
    },
}

# Page configuration
st.set_page_config(
    page_title="LLM Chat",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional dark theme CSS
st.markdown(
    """
    <style>
    /* Dark theme overrides */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid #2D3748;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #F7FAFC;
        font-weight: 600;
        font-size: 1.75rem;
        margin: 0;
    }
    .main-header p {
        color: #A0AEC0;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-ready {
        background-color: #276749;
        color: #C6F6D5;
    }
    .status-waiting {
        background-color: #744210;
        color: #FEFCBF;
    }
    
    /* Model info card */
    .model-card {
        background-color: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 6px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    .model-card code {
        color: #90CDF4;
        font-size: 0.75rem;
    }
    .model-card .label {
        color: #718096;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .model-card .value {
        color: #E2E8F0;
    }
    
    /* WandB link */
    .wandb-link {
        background-color: #2D3748;
        border: 1px solid #4A5568;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .wandb-link a {
        color: #F6AD55;
        text-decoration: none;
        font-weight: 500;
    }
    .wandb-link a:hover {
        text-decoration: underline;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        margin-bottom: 1rem;
    }
    .sidebar-section h3 {
        color: #A0AEC0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "messages": [],
        "conversation": ConversationHistory(
            system_prompt="You are a helpful, harmless, and honest AI assistant."
        ),
        "llm_engine": None,
        "model_loaded": False,
        "current_model": None,
        "metrics_logger": None,
        "wandb_url": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_chat() -> None:
    """Start a new chat session."""
    st.session_state.messages = []
    st.session_state.conversation = ConversationHistory(
        system_prompt="You are a helpful, harmless, and honest AI assistant."
    )


def render_sidebar() -> dict:
    """Render the sidebar with settings."""
    with st.sidebar:
        st.markdown("### Model")
        
        model_names = list(MODEL_PRESETS.keys())
        model_name = st.selectbox(
            "Select Model",
            options=model_names,
            index=0,
            disabled=st.session_state.model_loaded,
            label_visibility="collapsed",
        )
        
        model_info = MODEL_PRESETS[model_name]
        st.markdown(
            f'''<div class="model-card">
                <div class="label">Model ID</div>
                <code>{model_info["id"]}</code>
                <div class="label" style="margin-top:0.5rem">VRAM</div>
                <div class="value">{model_info["vram"]}</div>
                <div class="label" style="margin-top:0.5rem">Best for</div>
                <div class="value">{model_info["desc"]}</div>
            </div>''',
            unsafe_allow_html=True,
        )

        st.markdown("### GPU")
        gpu_memory = st.slider(
            "Memory Utilization",
            min_value=0.5,
            max_value=0.9,
            value=0.75,
            step=0.05,
            disabled=st.session_state.model_loaded,
        )
        
        context_length = st.select_slider(
            "Context Length",
            options=[1024, 2048, 4096, 8192],
            value=2048,
            disabled=st.session_state.model_loaded,
        )

        st.markdown("### Generation")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 128, 2048, 1024, 128)
        top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)

        st.markdown("### Monitoring")
        enable_wandb = st.checkbox("Enable WandB", value=True)

        st.divider()
        
        load_clicked = False
        if st.session_state.model_loaded:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Unload", use_container_width=True):
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
            with col2:
                if st.button("New Chat", use_container_width=True):
                    start_new_chat()
                    st.rerun()
        else:
            load_clicked = st.button("Load Model", use_container_width=True, type="primary")

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
    """Load the LLM model."""
    model_name = settings["model_name"]
    model_id = settings["model_id"]
    
    with st.spinner(f"Loading {model_name}..."):
        try:
            start_time = time.time()
            
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
            
            if st.session_state.metrics_logger:
                st.session_state.metrics_logger.log_model_load(load_time, model_name)
            
            st.session_state.llm_engine = engine
            st.session_state.model_loaded = True
            st.session_state.current_model = model_name
            
            st.success(f"Model loaded in {load_time:.1f}s")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            if st.session_state.metrics_logger:
                st.session_state.metrics_logger.end_session()


def render_chat_messages() -> None:
    """Render chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_response(settings: dict, user_input: str) -> None:
    """Generate and stream response."""
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
            
            latency = time.time() - start_time
            
            if st.session_state.metrics_logger:
                metrics = InferenceMetrics(
                    input_tokens=len(user_input.split()),
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
            response_placeholder.error(f"Error: {str(e)}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {str(e)}"}
            )


def main() -> None:
    """Main application."""
    initialize_session_state()

    # Header
    model_info = f" | {st.session_state.current_model}" if st.session_state.current_model else ""
    st.markdown(
        f'''<div class="main-header">
            <h1>LLM Chat</h1>
            <p>vLLM on NVIDIA A6000{model_info}</p>
        </div>''',
        unsafe_allow_html=True,
    )

    settings = render_sidebar()

    if settings["load_clicked"]:
        load_model(settings)

    # WandB link
    if st.session_state.wandb_url:
        st.markdown(
            f'<div class="wandb-link"><a href="{st.session_state.wandb_url}" target="_blank">View WandB Dashboard</a></div>',
            unsafe_allow_html=True,
        )

    # Status
    if st.session_state.model_loaded:
        st.markdown(
            f'<div class="status-badge status-ready">{st.session_state.current_model} Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-waiting">Select a model to begin</div>',
            unsafe_allow_html=True,
        )

    render_chat_messages()

    if prompt := st.chat_input("Type your message...", disabled=not st.session_state.model_loaded):
        generate_response(settings, prompt)


if __name__ == "__main__":
    main()

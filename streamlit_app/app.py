"""Streamlit chat application with shared global LLM engine."""

import os
import sys
import time

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_operations.llm_config import LLMConfig
from llm_operations.llm_inference import ConversationHistory, LLMEngine
from llm_operations.metrics import MetricsLogger, InferenceMetrics

# Fixed model configuration - Compatible with vLLM 0.4.2
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "Llama 3 8B Instruct"
GPU_MEMORY = 0.85
CONTEXT_LENGTH = 4096

# Page configuration
st.set_page_config(
    page_title="LLM Chat",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Professional dark theme CSS
st.markdown(
    """
    <style>
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
    .status-loading {
        background-color: #744210;
        color: #FEFCBF;
    }
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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_global_engine():
    """Load the LLM engine once and share across all users."""
    config = LLMConfig(
        model={
            "model_id": MODEL_ID,
            "gpu_memory_utilization": GPU_MEMORY,
            "max_model_len": CONTEXT_LENGTH,
            "trust_remote_code": True,
        }
    )
    engine = LLMEngine(config)
    engine.load_model()
    return engine


@st.cache_resource(show_spinner=False)
def get_global_metrics():
    """Initialize global metrics logger."""
    metrics = MetricsLogger(project="llm-chat")
    metrics.start_session(
        model_name=MODEL_NAME,
        model_id=MODEL_ID,
        gpu_memory_utilization=GPU_MEMORY,
        context_length=CONTEXT_LENGTH,
    )
    return metrics


def initialize_session_state() -> None:
    """Initialize per-user session state (conversation only)."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationHistory(
            system_prompt="You are a helpful, harmless, and honest AI assistant."
        )


def start_new_chat() -> None:
    """Start a new chat session."""
    st.session_state.messages = []
    st.session_state.conversation = ConversationHistory(
        system_prompt="You are a helpful, harmless, and honest AI assistant."
    )


def render_sidebar() -> dict:
    """Render simplified sidebar with generation params only."""
    with st.sidebar:
        st.markdown("### Generation")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 128, 2048, 1024, 128)
        top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)

        st.divider()
        
        if st.button("New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }


def render_chat_messages() -> None:
    """Render chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_response(engine: LLMEngine, metrics: MetricsLogger, settings: dict, user_input: str) -> None:
    """Generate and stream response using shared engine."""
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        start_time = time.time()
        token_count = 0

        try:
            for token in engine.generate_stream(
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
            
            if metrics:
                inference_metrics = InferenceMetrics(
                    input_tokens=len(user_input.split()),
                    output_tokens=token_count,
                    latency_seconds=latency,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                )
                inference_metrics.calculate_tokens_per_second()
                metrics.log_inference(inference_metrics)
                metrics.log_gpu_snapshot()
            
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
    st.markdown(
        f'''<div class="main-header">
            <h1>LLM Chat</h1>
            <p>{MODEL_NAME} on NVIDIA A6000</p>
        </div>''',
        unsafe_allow_html=True,
    )

    settings = render_sidebar()

    # Load global engine (cached - only loads once)
    with st.spinner(f"Loading {MODEL_NAME}... This only happens once on startup."):
        try:
            engine = get_global_engine()
            metrics = get_global_metrics()
            model_ready = True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            model_ready = False
            engine = None
            metrics = None

    # Status
    if model_ready:
        st.markdown(
            f'<div class="status-badge status-ready">{MODEL_NAME} Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-loading">Model loading failed</div>',
            unsafe_allow_html=True,
        )
        return

    # WandB link
    if metrics and metrics.run_url:
        st.markdown(
            f'<div class="wandb-link"><a href="{metrics.run_url}" target="_blank">View WandB Dashboard</a></div>',
            unsafe_allow_html=True,
        )

    render_chat_messages()

    if prompt := st.chat_input("Type your message..."):
        generate_response(engine, metrics, settings, prompt)


if __name__ == "__main__":
    main()

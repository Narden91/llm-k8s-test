"""Streamlit chat application with shared global LLM engine - Optimized."""

import os
import sys
import time

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_operations.llm_config import LLMConfig
from llm_operations.llm_inference import ConversationHistory, LLMEngine
from llm_operations.metrics import MetricsLogger, InferenceMetrics

# Configuration
GPU_MEMORY = 0.85
CONTEXT_LENGTH = 4096
FALLBACK_MODELS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama 3 8B"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B"),
]

# Page config (must be first Streamlit call)
st.set_page_config(page_title="LLM Chat", layout="wide", initial_sidebar_state="collapsed")

# CSS cached as constant (avoid re-parsing)
_CSS = """
<style>
.main-header{text-align:center;padding:1rem 0;border-bottom:1px solid #2D3748;margin-bottom:1rem}
.main-header h1{color:#F7FAFC;font-weight:600;font-size:1.5rem;margin:0}
.main-header p{color:#A0AEC0;font-size:0.85rem;margin:0.3rem 0 0}
.status-badge{display:inline-block;padding:0.3rem 0.8rem;border-radius:4px;font-size:0.8rem;margin-bottom:0.5rem}
.status-ready{background:#276749;color:#C6F6D5}
.status-error{background:#742a2a;color:#FED7D7}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_engine():
    """Load LLM engine with fallback. Cached globally."""
    for model_id, name in FALLBACK_MODELS:
        try:
            cfg = LLMConfig(model={
                "model_id": model_id,
                "gpu_memory_utilization": GPU_MEMORY,
                "max_model_len": CONTEXT_LENGTH,
                "trust_remote_code": True,
            })
            engine = LLMEngine(cfg)
            engine.load_model()
            return engine, name
        except Exception as e:
            if "gated" in str(e).lower() or "403" in str(e):
                continue
            raise
    raise RuntimeError("All models failed to load")


@st.cache_resource(show_spinner=False)
def load_metrics(model_name: str):
    """Initialize metrics. Cached globally."""
    m = MetricsLogger(project="llm-chat")
    m.start_session(model_name=model_name, model_id="", gpu_memory_utilization=GPU_MEMORY, context_length=CONTEXT_LENGTH)
    return m


def init_state():
    """Initialize session state once."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.conversation = ConversationHistory(
            system_prompt="You are a helpful AI assistant."
        )


def new_chat():
    """Reset chat state."""
    st.session_state.messages = []
    st.session_state.conversation = ConversationHistory(
        system_prompt="You are a helpful AI assistant."
    )


def stream_response(engine, prompt: str, temp: float, max_tok: int, top_p: float):
    """Stream response with accurate token metrics."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        chunks = []
        t0 = time.perf_counter()
        request_output = None
        
        try:
            # Add user message to conversation before calling generate_stream
            st.session_state.conversation.add_user_message(prompt)
            
            for token, output in engine.generate_stream(
                prompt=prompt,
                conversation=st.session_state.conversation,
                temperature=temp,
                max_tokens=max_tok,
                top_p=top_p,
            ):
                chunks.append(token)
                request_output = output
                # Batch updates every 5 tokens for efficiency
                if len(chunks) % 5 == 0:
                    placeholder.markdown("".join(chunks) + "▌")
            
            response = "".join(chunks)
            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Extract accurate token counts from RequestOutput
            latency = time.perf_counter() - t0
            if request_output:
                # Get prompt token count from RequestOutput
                input_tokens = len(request_output.prompt_token_ids)
                # Get output token count from CompletionOutput
                output_tokens = len(request_output.outputs[0].token_ids)
            else:
                # Fallback if no output (shouldn't happen)
                input_tokens = 0
                output_tokens = 0
            
            return input_tokens, output_tokens, latency
            
        except Exception as e:
            placeholder.error(f"Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            return 0, 0, 0


def main():
    init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    
    # Sidebar (minimal)
    with st.sidebar:
        st.markdown("### Settings")
        temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_tok = st.slider("Max Tokens", 256, 2048, 1024, 256)
        top_p = st.slider("Top P", 0.5, 1.0, 0.9, 0.05)
        st.divider()
        if st.button("New Chat", use_container_width=True):
            new_chat()
            st.rerun()
    
    # Load model (cached)
    try:
        engine, model_name = load_engine()
        metrics = load_metrics(model_name)
        ready = True
    except Exception as e:
        st.error(f"Failed: {e}")
        ready = False
        model_name = "Error"
    
    # Header
    st.markdown(f'<div class="main-header"><h1>LLM Chat</h1><p>{model_name}</p></div>', unsafe_allow_html=True)
    
    if ready:
        st.markdown(f'<div class="status-badge status-ready">{model_name} Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-error">Load Failed</div>', unsafe_allow_html=True)
        return
    
    # Render messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Message..."):
        input_tokens, output_tokens, latency = stream_response(engine, prompt, temp, max_tok, top_p)
        if metrics and output_tokens > 0:
            m = InferenceMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=latency,
                temperature=temp,
                max_tokens=max_tok,
            )
            m.calculate_tokens_per_second()
            metrics.log_inference(m)


if __name__ == "__main__":
    main()

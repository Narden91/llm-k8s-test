"""Streamlit chat application for LLM interaction."""

import os
import sys

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_operations.llm_config import LLMConfig
from llm_operations.llm_inference import ConversationHistory, LLMEngine

# Page configuration
st.set_page_config(
    page_title="LLM Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better chat styling
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-loading {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
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
            system_prompt="You are a helpful, harmless, and honest AI assistant. "
            "Provide clear, accurate, and thoughtful responses."
        )

    if "llm_engine" not in st.session_state:
        st.session_state.llm_engine = None

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


def render_sidebar() -> dict:
    """Render the sidebar with model settings.

    Returns:
        Dictionary of generation parameters.
    """
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Model Configuration")
        model_id = st.text_input(
            "Model ID",
            value="mistralai/Mistral-7B-Instruct-v0.3",
            help="Hugging Face model identifier",
        )

        gpu_memory = st.slider(
            "GPU Memory Utilization",
            min_value=0.5,
            max_value=0.95,
            value=0.9,
            step=0.05,
            help="Fraction of GPU memory to allocate",
        )

        st.divider()
        st.subheader("Generation Parameters")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative, lower = more focused",
        )

        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=4096,
            value=2048,
            step=128,
            help="Maximum length of generated response",
        )

        top_p = st.slider(
            "Top P (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )

        st.divider()
        st.subheader("Actions")

        load_clicked = st.button(
            "🚀 Load Model",
            disabled=st.session_state.model_loaded,
            use_container_width=True,
        )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation.clear()
            st.rerun()

        return {
            "model_id": model_id,
            "gpu_memory": gpu_memory,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "load_clicked": load_clicked,
        }


def load_model(model_id: str, gpu_memory: float) -> None:
    """Load the LLM model.

    Args:
        model_id: Hugging Face model identifier.
        gpu_memory: GPU memory utilization fraction.
    """
    with st.spinner("Loading model... This may take a few minutes."):
        config = LLMConfig(
            model={"model_id": model_id, "gpu_memory_utilization": gpu_memory}
        )
        engine = LLMEngine(config)
        engine.load_model()
        st.session_state.llm_engine = engine
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")
        st.rerun()


def render_chat_messages() -> None:
    """Render the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_response(
    user_input: str, temperature: float, max_tokens: int, top_p: float
) -> None:
    """Generate and stream the assistant's response.

    Args:
        user_input: The user's message.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
    """
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for token in st.session_state.llm_engine.generate_stream(
                prompt=user_input,
                conversation=st.session_state.conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            ):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
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
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 LLM Chat Assistant</h1>
            <p>Powered by Mistral 7B on NVIDIA A6000</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar and settings
    settings = render_sidebar()

    # Load model if requested
    if settings["load_clicked"]:
        load_model(settings["model_id"], settings["gpu_memory"])

    # Model status indicator
    if st.session_state.model_loaded:
        st.markdown(
            '<div class="status-indicator status-ready">✅ Model Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-indicator status-loading">⏳ Model Not Loaded - Click "Load Model" in sidebar</div>',
            unsafe_allow_html=True,
        )

    # Chat messages
    render_chat_messages()

    # Chat input
    if prompt := st.chat_input(
        "Type your message...", disabled=not st.session_state.model_loaded
    ):
        generate_response(
            prompt,
            settings["temperature"],
            settings["max_tokens"],
            settings["top_p"],
        )


if __name__ == "__main__":
    main()

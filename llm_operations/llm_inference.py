"""LLM inference engine using vLLM for high-performance serving."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from llm_operations.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class ConversationMessage:
    """Represents a single message in a conversation."""

    def __init__(self, role: str, content: str) -> None:
        """Initialize a conversation message.

        Args:
            role: Either 'user' or 'assistant'.
            content: The message content.
        """
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    """Manages conversation history for multi-turn chat."""

    def __init__(self, system_prompt: str | None = None) -> None:
        """Initialize conversation history.

        Args:
            system_prompt: Optional system prompt to prepend to conversations.
        """
        self.system_prompt = system_prompt
        self.messages: list[ConversationMessage] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append(ConversationMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.messages.append(ConversationMessage(role="assistant", content=content))

    def format_for_mistral(self) -> str:
        """Format conversation history for Mistral Instruct model.

        Returns:
            Formatted prompt string for Mistral.
        """
        prompt_parts = []

        if self.system_prompt:
            prompt_parts.append(f"<s>[INST] {self.system_prompt}\n\n")
        else:
            prompt_parts.append("<s>[INST] ")

        for i, msg in enumerate(self.messages):
            if msg.role == "user":
                if i > 0:
                    prompt_parts.append("[INST] ")
                prompt_parts.append(f"{msg.content} [/INST]")
            else:
                prompt_parts.append(f" {msg.content}</s>")

        return "".join(prompt_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []


class LLMEngine:
    """High-performance LLM inference engine using vLLM."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the LLM engine.

        Args:
            config: LLM configuration. Uses defaults if None.
        """
        self.config = config or LLMConfig()
        self._llm: LLM | None = None

    def load_model(self) -> None:
        """Load the LLM model into memory.

        This should be called once at startup. Model loading may take
        several minutes for large models.
        """
        logger.info(f"Loading model: {self.config.model.model_id}")

        # Use max_model_len to limit context and reduce memory usage
        max_len = self.config.model.max_model_len or 4096  # Default to 4k context
        
        self._llm = LLM(
            model=self.config.model.model_id,
            revision=self.config.model.revision,
            dtype=self.config.model.dtype,
            gpu_memory_utilization=self.config.model.gpu_memory_utilization,
            max_model_len=max_len,
            trust_remote_code=self.config.model.trust_remote_code,
            tensor_parallel_size=self.config.tensor_parallel_size,
        )

        logger.info("Model loaded successfully.")

    def _get_sampling_params(self, **overrides) -> SamplingParams:
        """Create sampling parameters from config with optional overrides.

        Args:
            **overrides: Override specific generation parameters.

        Returns:
            SamplingParams configured for generation.
        """
        gen_config = self.config.generation
        return SamplingParams(
            max_tokens=overrides.get("max_tokens", gen_config.max_tokens),
            temperature=overrides.get("temperature", gen_config.temperature),
            top_p=overrides.get("top_p", gen_config.top_p),
            top_k=overrides.get("top_k", gen_config.top_k),
            repetition_penalty=overrides.get(
                "repetition_penalty", gen_config.repetition_penalty
            ),
            stop=overrides.get("stop", gen_config.stop_sequences),
        )

    def generate(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **generation_kwargs,
    ) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The user's input prompt.
            conversation: Optional conversation history for context.
            **generation_kwargs: Override generation parameters.

        Returns:
            Generated text response.

        Raises:
            RuntimeError: If model hasn't been loaded.
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build full prompt with conversation context
        if conversation is not None:
            conversation.add_user_message(prompt)
            full_prompt = conversation.format_for_mistral()
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"

        sampling_params = self._get_sampling_params(**generation_kwargs)
        outputs: list[RequestOutput] = self._llm.generate(
            [full_prompt], sampling_params
        )

        generated_text = outputs[0].outputs[0].text.strip()

        # Update conversation history with response
        if conversation is not None:
            conversation.add_assistant_message(generated_text)

        return generated_text

    def generate_stream(
        self,
        prompt: str,
        conversation: ConversationHistory | None = None,
        **generation_kwargs,
    ) -> Iterator[str]:
        """Generate a streaming response for the given prompt.

        Args:
            prompt: The user's input prompt.
            conversation: Optional conversation history for context.
            **generation_kwargs: Override generation parameters.

        Yields:
            Generated text tokens as they become available.

        Raises:
            RuntimeError: If model hasn't been loaded.
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build full prompt with conversation context
        if conversation is not None:
            conversation.add_user_message(prompt)
            full_prompt = conversation.format_for_mistral()
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"

        sampling_params = self._get_sampling_params(**generation_kwargs)

        # Use streaming generation
        full_response = ""
        for output in self._llm.generate([full_prompt], sampling_params, use_tqdm=False):
            token = output.outputs[0].text[len(full_response):]
            full_response = output.outputs[0].text
            if token:
                yield token

        # Update conversation history with complete response
        if conversation is not None:
            conversation.add_assistant_message(full_response.strip())

    def is_loaded(self) -> bool:
        """Check if the model is loaded.

        Returns:
            True if model is loaded and ready for inference.
        """
        return self._llm is not None

    def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None

            # Force CUDA memory cleanup
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded.")

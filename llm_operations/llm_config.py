"""LLM configuration using Pydantic for validation."""

from typing import Literal

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration for text generation parameters."""

    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values increase randomness.",
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold.",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        description="Top-k sampling parameter.",
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens.",
    )
    stop_sequences: list[str] = Field(
        default_factory=lambda: ["</s>", "[/INST]"],
        description="Sequences that signal end of generation.",
    )


class ModelConfig(BaseModel):
    """Configuration for the LLM model."""
    
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="Hugging Face model identifier.",
    )
    revision: str = Field(
        default="main",
        description="Model revision/branch to use.",
    )
    dtype: Literal["float16", "bfloat16", "auto"] = Field(
        default="auto",
        description="Data type for model weights.",
    )
    gpu_memory_utilization: float = Field(
        default=0.8,
        ge=0.1,
        le=0.99,
        description="Fraction of GPU memory to use for model.",
    )
    max_model_len: int | None = Field(
        default=4096,
        description="Maximum context length. Lower values reduce memory usage.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code from HF hub.",
    )


class LLMConfig(BaseModel):
    """Complete LLM configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism.",
    )

    @classmethod
    def from_yaml(cls, path: str) -> "LLMConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

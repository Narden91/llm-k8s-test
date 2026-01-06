"""Metrics logging module using Weights & Biases."""

import time
import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import wandb, gracefully handle if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Metrics logging disabled.")


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""
    
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    tokens_per_second: float = 0.0
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    
    def calculate_tokens_per_second(self) -> None:
        """Calculate tokens per second from output tokens and latency."""
        if self.latency_seconds > 0:
            self.tokens_per_second = self.output_tokens / self.latency_seconds


class MetricsLogger:
    """WandB-based metrics logger for LLM inference."""
    
    def __init__(
        self,
        project: str = "llm-chat",
        enabled: bool = True,
    ) -> None:
        """Initialize metrics logger.
        
        Args:
            project: WandB project name.
            enabled: Whether to enable logging.
        """
        self.project = project
        self.enabled = enabled and WANDB_AVAILABLE
        self._run = None
        self._request_count = 0
        self._total_tokens = 0
        self._start_time: float | None = None
        
    def start_session(
        self,
        model_name: str,
        model_id: str,
        gpu_memory_utilization: float,
        **config: Any,
    ) -> str | None:
        """Start a new WandB session.
        
        Args:
            model_name: Display name of the model.
            model_id: HuggingFace model ID.
            gpu_memory_utilization: GPU memory fraction.
            **config: Additional configuration.
            
        Returns:
            WandB run URL or None if disabled.
        """
        if not self.enabled:
            return None
            
        try:
            self._run = wandb.init(
                project=self.project,
                config={
                    "model_name": model_name,
                    "model_id": model_id,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    **config,
                },
                reinit=True,
            )
            self._start_time = time.time()
            logger.info(f"WandB session started: {self._run.url}")
            return self._run.url
        except Exception as e:
            logger.error(f"Failed to start WandB session: {e}")
            self.enabled = False
            return None
    
    def log_model_load(self, load_time_seconds: float, model_name: str) -> None:
        """Log model loading metrics.
        
        Args:
            load_time_seconds: Time taken to load the model.
            model_name: Name of the loaded model.
        """
        if not self.enabled or not self._run:
            return
            
        try:
            wandb.log({
                "model/load_time_seconds": load_time_seconds,
                "model/name": model_name,
            })
        except Exception as e:
            logger.error(f"Failed to log model load: {e}")
    
    def log_inference(self, metrics: InferenceMetrics) -> None:
        """Log inference metrics for a single request.
        
        Args:
            metrics: InferenceMetrics dataclass.
        """
        if not self.enabled or not self._run:
            return
            
        self._request_count += 1
        self._total_tokens += metrics.output_tokens
        
        try:
            wandb.log({
                "inference/input_tokens": metrics.input_tokens,
                "inference/output_tokens": metrics.output_tokens,
                "inference/latency_seconds": metrics.latency_seconds,
                "inference/tokens_per_second": metrics.tokens_per_second,
                "inference/temperature": metrics.temperature,
                "inference/max_tokens": metrics.max_tokens,
                "session/request_count": self._request_count,
                "session/total_tokens": self._total_tokens,
            })
        except Exception as e:
            logger.error(f"Failed to log inference: {e}")
    
    def log_gpu_snapshot(self) -> None:
        """Log GPU memory snapshot (for manual GPU logging)."""
        if not self.enabled or not self._run:
            return
            
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                wandb.log({
                    "gpu/memory_allocated_gb": memory_allocated,
                    "gpu/memory_reserved_gb": memory_reserved,
                })
        except Exception as e:
            logger.error(f"Failed to log GPU snapshot: {e}")
    
    def end_session(self) -> None:
        """End the WandB session gracefully."""
        if not self.enabled or not self._run:
            return
            
        try:
            session_duration = time.time() - self._start_time if self._start_time else 0
            wandb.log({
                "session/duration_seconds": session_duration,
                "session/total_requests": self._request_count,
                "session/total_tokens_generated": self._total_tokens,
            })
            wandb.finish()
            self._run = None
            logger.info("WandB session ended.")
        except Exception as e:
            logger.error(f"Failed to end WandB session: {e}")
    
    @property
    def run_url(self) -> str | None:
        """Get the WandB run URL."""
        if self._run:
            return self._run.url
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if a WandB session is active."""
        return self._run is not None


# Global metrics logger instance
_metrics_logger: MetricsLogger | None = None


def get_metrics_logger() -> MetricsLogger:
    """Get or create the global metrics logger."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger

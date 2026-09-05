"""
Pydantic Models for Sentiment Classification API.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PredictRequest(BaseSchema):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text content for sentiment analysis",
        examples=["I absolutely love using this clean and fast machine learning model!"]
    )
    model: Optional[str] = Field(
        default="sklearn-logreg",
        description="Model identifier to use for inference"
    )


class ProbabilityScores(BaseSchema):
    negative: float = Field(..., description="Probability of negative sentiment [0, 1]")
    positive: float = Field(..., description="Probability of positive sentiment [0, 1]")


class PredictResponse(BaseSchema):
    input: str = Field(..., description="Sanitized input text preview")
    prediction: int = Field(..., description="Binary sentiment: 0 (Negative) or 1 (Positive)")
    sentiment: str = Field(..., description="Sentiment label: 'Positive' or 'Negative'")
    confidence: float = Field(..., description="Prediction confidence score between 0.5 and 1.0")
    probabilities: ProbabilityScores = Field(..., description="Detailed class probabilities")
    inference_time_ms: float = Field(..., description="Inference execution latency in milliseconds")
    model_used: str = Field(..., description="Name of the model utilized")
    api_version: str = Field(default="2.0.0", description="API release version")


class ModelInfoResponse(BaseSchema):
    available_models: List[str]
    default_model: str
    model_type: str
    pipeline_steps: List[str]
    features_count: int
    solver: str
    version: str


class HealthResponse(BaseSchema):
    status: str
    model_loaded: bool
    model_path: str
    version: str
    uptime_seconds: float
    timestamp: str


class ResearchBenchmarkResponse(BaseSchema):
    model_name: str
    dataset_name: str
    total_training_samples: int
    test_samples: int
    vocabulary_features: int
    metrics: Dict[str, Any]
    benchmarks_p95_latency_ms: float
    throughput_inferences_per_sec: float


class ErrorResponse(BaseSchema):
    error: bool = True
    status_code: int
    message: str
    timestamp: str

"""
Application Configuration and Environment Variable Management.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    APP_NAME: str = "Binary Sentiment Classification API"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # CORS settings
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: [
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ])

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PREDICT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PREDICT_PER_MINUTE", "60"))
    RATE_LIMIT_GENERAL_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_GENERAL_PER_MINUTE", "120"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/sentiment_analyzer.log")
    LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # Model configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "")

    def get_resolved_model_path(self) -> str:
        """Finds the model file across common relative and absolute locations."""
        if self.MODEL_PATH and os.path.exists(self.MODEL_PATH):
            return self.MODEL_PATH

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            # In backend/models/sklearn/
            os.path.join(current_dir, "..", "models", "sklearn", "logreg-80k.joblib"),
            # In root models/sklearn/
            os.path.join(current_dir, "..", "..", "models", "sklearn", "logreg-80k.joblib"),
            # Direct relative
            "models/sklearn/logreg-80k.joblib",
            "backend/models/sklearn/logreg-80k.joblib",
        ]
        for candidate in candidates:
            resolved = os.path.abspath(candidate)
            if os.path.exists(resolved):
                return resolved

        # Fallback to candidates[0] even if missing (will report error in classifier service)
        return os.path.abspath(candidates[0])


settings = Settings()

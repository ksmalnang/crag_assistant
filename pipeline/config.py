"""Application settings and configuration management."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", description="Embedding model"
    )

    # Qdrant Vector Store
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_collection: str = Field(
        default="documents", description="Qdrant collection name"
    )
    qdrant_vector_size: int = Field(default=3072, description="Vector dimension size")

    # LangSmith Tracing (Optional)
    langchain_api_key: Optional[str] = Field(
        default=None, description="LangSmith API key"
    )
    langchain_tracing_v2: bool = Field(
        default=False, description="Enable LangSmith tracing"
    )
    langchain_project: str = Field(
        default="crag-pipeline-dev", description="LangSmith project name"
    )

    # Application Settings
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment name")

    # Ingestion Settings
    max_file_size_bytes: int = Field(
        default=52_428_800,  # 50MB
        description="Maximum allowed file size in bytes (default: 50MB)",
    )
    watch_dir: str = Field(
        default="data/watch",
        description="Directory to watch for file ingestion",
    )

    # Parsing Settings (Docling)
    parser_timeout_seconds: int = Field(
        default=120,
        description="Timeout per file for Docling parsing (default: 120s)",
    )
    export_page_images: bool = Field(
        default=False,
        description="Export page images as PNG for source preview",
    )
    page_image_dpi: int = Field(
        default=150,
        description="DPI for exported page images",
    )
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR for scanned PDFs",
    )
    ocr_confidence_threshold: float = Field(
        default=0.6,
        description="Minimum OCR confidence threshold (below this triggers low_confidence flag)",
    )
    scanned_pdf_char_threshold: int = Field(
        default=100,
        description="Average chars per page below this threshold flags PDF as scanned",
    )

    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is properly configured."""
        return bool(self.langchain_api_key and self.langchain_tracing_v2)


# Global settings instance
settings = Settings()

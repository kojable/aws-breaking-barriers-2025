"""
Bedrock Configuration for Amazon Bedrock AgentCore
Self-Healing RAN MVP with Strands Agents
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables from .env file
load_dotenv()


class BedrockConfig(BaseSettings):
    """
    Configuration for Amazon Bedrock AgentCore integration.
    
    All settings can be configured via environment variables with BEDROCK_ prefix
    or through a .env file in the project root.
    
    Example .env file:
        AWS_REGION=us-east-1
        BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
        BEDROCK_MAX_TOKENS=4096
    """
    
    # ========================================================================
    # AWS Configuration
    # ========================================================================
    
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock service"
    )
    
    aws_profile: Optional[str] = Field(
        default=None,
        description="AWS profile name (optional)"
    )
    
    # ========================================================================
    # Bedrock Model Configuration
    # ========================================================================
    
    # Default to Amazon Nova Lite (matches your notebook)
    # To use Claude instead, change to: anthropic.claude-3-5-sonnet-20241022-v2:0
    model_id: str = Field(
        default="us.amazon.nova-lite-v1:0",
        description="Primary Bedrock model ID (Nova Lite or Claude)"
    )
    
    model_id_fast: str = Field(
        default="us.amazon.nova-lite-v1:0",
        description="Fast model for simple tasks"
    )
    
    model_id_haiku: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0",
        description="Claude Haiku (alternative to Nova)"
    )
    
    model_id_embeddings: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Embeddings model for semantic search"
    )
    
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens in model response"
    )
    
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0=deterministic, 1=creative)"
    )
    
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    # ========================================================================
    # Agent Configuration
    # ========================================================================
    
    agent_timeout: int = Field(
        default=300,
        ge=1,
        description="Agent execution timeout in seconds"
    )
    
    agent_max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed API calls"
    )
    
    agent_retry_delay: int = Field(
        default=2,
        ge=1,
        description="Initial retry delay in seconds (exponential backoff)"
    )
    
    agent_max_delay: int = Field(
        default=60,
        ge=1,
        description="Maximum retry delay in seconds"
    )
    
    # ========================================================================
    # Data Configuration
    # ========================================================================
    
    data_path: Path = Field(
        default=Path("cell_kpi_anomalies.csv"),
        description="Path to input data file"
    )
    
    output_path: Path = Field(
        default=Path("results"),
        description="Directory for output files"
    )
    
    cache_path: Path = Field(
        default=Path(".cache"),
        description="Directory for cached data"
    )
    
    # ========================================================================
    # Logging Configuration
    # ========================================================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    log_file: Path = Field(
        default=Path("logs/agent_execution.log"),
        description="Path to log file"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # ========================================================================
    # Feature Flags
    # ========================================================================
    
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching"
    )
    
    enable_monitoring: bool = Field(
        default=True,
        description="Enable CloudWatch monitoring"
    )
    
    enable_streaming: bool = Field(
        default=False,
        description="Enable streaming responses (not fully implemented)"
    )
    
    # ========================================================================
    # Performance Tuning
    # ========================================================================
    
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Batch size for data processing"
    )
    
    parallel_agents: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of parallel agent executions"
    )
    
    # ========================================================================
    # Pydantic Configuration
    # ========================================================================
    
    class Config:
        env_file = ".env"
        env_prefix = "BEDROCK_"
        case_sensitive = False
        extra = "allow"
    
    # ========================================================================
    # Model Family Detection
    # ========================================================================
    
    @property
    def model_family(self) -> str:
        """
        Detect model family from model_id for API format selection
        
        Returns:
            'claude', 'nova', 'titan', or 'unknown'
        """
        model_lower = self.model_id.lower()
        if 'anthropic' in model_lower or 'claude' in model_lower:
            return 'claude'
        elif 'nova' in model_lower:
            return 'nova'
        elif 'titan' in model_lower:
            return 'titan'
        else:
            return 'unknown'
    
    # ========================================================================
    # Post-initialization Setup
    # ========================================================================
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.output_path,
            self.cache_path,
            self.log_file.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_model_config(self, model_type: str = "primary") -> dict:
        """
        Get model configuration dictionary for Bedrock API calls
        
        Args:
            model_type: Type of model ('primary', 'haiku', 'embeddings')
        
        Returns:
            Dictionary with model configuration
        """
        model_map = {
            "primary": self.model_id,
            "haiku": self.model_id_haiku,
            "embeddings": self.model_id_embeddings
        }
        
        return {
            "modelId": model_map.get(model_type, self.model_id),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    
    def validate_aws_credentials(self) -> bool:
        """
        Validate AWS credentials are configured
        
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            import boto3
            sts = boto3.client('sts', region_name=self.aws_region)
            sts.get_caller_identity()
            return True
        except Exception:
            return False
    
    def get_bedrock_client(self, runtime: bool = True):
        """
        Get Boto3 Bedrock client
        
        Args:
            runtime: If True, return bedrock-runtime client, else bedrock client
        
        Returns:
            Boto3 Bedrock client
        """
        import boto3
        
        service_name = 'bedrock-runtime' if runtime else 'bedrock'
        
        session_kwargs = {'region_name': self.aws_region}
        if self.aws_profile:
            session_kwargs['profile_name'] = self.aws_profile
        
        session = boto3.Session(**session_kwargs)
        return session.client(service_name)


# ============================================================================
# Global Configuration Instance
# ============================================================================

config = BedrockConfig()


# ============================================================================
# Configuration Validator
# ============================================================================

def validate_configuration() -> tuple[bool, list[str]]:
    """
    Validate the complete configuration
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check AWS credentials
    if not config.validate_aws_credentials():
        errors.append("AWS credentials not configured or invalid")
    
    # Check data file exists
    if not config.data_path.exists():
        errors.append(f"Data file not found: {config.data_path}")
    
    # Check model IDs are valid format
    if not config.model_id.startswith("anthropic."):
        errors.append(f"Invalid primary model ID: {config.model_id}")
    
    # Check log level is valid
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.log_level.upper() not in valid_levels:
        errors.append(f"Invalid log level: {config.log_level}")
    
    return len(errors) == 0, errors


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Bedrock AgentCore Configuration")
    print("=" * 70)
    print()
    print(f"AWS Region:        {config.aws_region}")
    print(f"Primary Model:     {config.model_id}")
    print(f"Fast Model:        {config.model_id_haiku}")
    print(f"Max Tokens:        {config.max_tokens}")
    print(f"Temperature:       {config.temperature}")
    print(f"Data Path:         {config.data_path}")
    print(f"Output Path:       {config.output_path}")
    print(f"Log Level:         {config.log_level}")
    print()
    
    # Validate configuration
    is_valid, errors = validate_configuration()
    
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in errors:
            print(f"   - {error}")
    
    print()
    print("=" * 70)

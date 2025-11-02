"""
Base Agent Class for Amazon Bedrock AgentCore with Strands SDK
Self-Healing RAN MVP
"""
import json
import time
import logging
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod

import boto3
from botocore.exceptions import ClientError

try:
    from bedrock_config import config
except ImportError:
    # Fallback if config not available
    class config:
        aws_region = "us-east-1"
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        max_tokens = 4096
        temperature = 0.3
        agent_max_retries = 5
        agent_retry_delay = 2
        agent_max_delay = 60
        log_level = "INFO"

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BedrockAgent(ABC):
    """
    Base Agent class integrating with Amazon Bedrock AgentCore.
    
    This class provides:
    - Bedrock model invocation with retry logic
    - Throttling protection with exponential backoff
    - Structured logging
    - Error handling and recovery
    - Tool/function calling support
    
    Subclasses should implement:
    - __init__(): Initialize agent-specific attributes
    - execute(): Main agent logic
    - Any @tool decorated methods for agent capabilities
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        model_id: Optional[str] = None
    ):
        """
        Initialize the Bedrock Agent
        
        Args:
            name: Agent name (e.g., "IngestionAgent")
            description: Brief description of agent capabilities
            instructions: Detailed instructions for the agent's behavior
            model_id: Optional specific model ID to use
        """
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model_id = model_id or config.model_id
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime',
            region_name=config.aws_region
        )
        
        # Setup logging
        self.logger = logging.getLogger(f"Agent.{name}")
        
        # Agent state
        self.tools = []
        self.conversation_history = []
        
        self.logger.info(f"Initialized {name} with model {self.model_id}")
    
    def invoke_model(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Invoke Bedrock model with automatic retry and throttling protection.
        
        Args:
            prompt: User prompt/message
            max_tokens: Maximum tokens in response (defaults to config)
            temperature: Sampling temperature (defaults to config)
            system_prompt: Optional system prompt override
        
        Returns:
            Model response text
        
        Raises:
            Exception: If all retries are exhausted
        """
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature
        system_prompt = system_prompt or self.instructions
        
        # Prepare request body for Claude models
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Add system prompt if provided
        if system_prompt:
            body["system"] = system_prompt
        
        # Retry loop with exponential backoff
        for attempt in range(config.agent_max_retries):
            try:
                self.logger.debug(f"Invoking model (attempt {attempt + 1}/{config.agent_max_retries})")
                
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                response_text = response_body['content'][0]['text']
                
                self.logger.debug(f"Model response received ({len(response_text)} chars)")
                
                # Store in conversation history
                self.conversation_history.append({
                    'prompt': prompt,
                    'response': response_text,
                    'timestamp': time.time()
                })
                
                return response_text
            
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                # Handle throttling
                if error_code == 'ThrottlingException' and attempt < config.agent_max_retries - 1:
                    delay = min(
                        config.agent_retry_delay * (2 ** attempt),
                        config.agent_max_delay
                    )
                    self.logger.warning(
                        f"Throttled by Bedrock. Retrying in {delay}s... "
                        f"(attempt {attempt + 1}/{config.agent_max_retries})"
                    )
                    time.sleep(delay)
                    continue
                
                # Handle other errors
                self.logger.error(f"Bedrock API error: {error_code} - {error_message}")
                
                if attempt < config.agent_max_retries - 1:
                    delay = config.agent_retry_delay * (2 ** attempt)
                    self.logger.warning(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            
            except Exception as e:
                self.logger.error(f"Unexpected error invoking model: {str(e)}")
                
                if attempt < config.agent_max_retries - 1:
                    delay = config.agent_retry_delay
                    self.logger.warning(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        
        raise Exception(
            f"Failed to invoke model after {config.agent_max_retries} attempts"
        )
    
    def invoke_with_json_response(
        self,
        prompt: str,
        schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Invoke model and parse JSON response.
        
        Args:
            prompt: User prompt requesting JSON output
            schema: Optional JSON schema for validation
        
        Returns:
            Parsed JSON dictionary
        """
        # Enhance prompt to request JSON
        json_prompt = f"""
{prompt}

Please provide your response as a valid JSON object. Ensure proper formatting.
"""
        
        response_text = self.invoke_model(json_prompt)
        
        # Extract JSON from response (handle code blocks)
        json_text = response_text
        if '```json' in response_text:
            json_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            json_text = response_text.split('```')[1].split('```')[0].strip()
        
        try:
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response text: {response_text}")
            raise
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Main execution method - must be implemented by subclasses.
        
        Args:
            input_data: Input data for the agent
        
        Returns:
            Dictionary with execution results
        """
        pass
    
    def register_tool(self, func, name: Optional[str] = None, description: Optional[str] = None):
        """
        Register a tool/function for this agent.
        
        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or "No description"
        
        self.tools.append({
            'name': tool_name,
            'description': tool_description,
            'function': func
        })
        
        self.logger.info(f"Registered tool: {tool_name}")
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation history.
        
        Returns:
            Formatted conversation summary
        """
        if not self.conversation_history:
            return "No conversation history"
        
        summary = f"Conversation History ({len(self.conversation_history)} exchanges):\n"
        for i, exchange in enumerate(self.conversation_history[-5:], 1):
            summary += f"\n{i}. Prompt: {exchange['prompt'][:100]}..."
            summary += f"\n   Response: {exchange['response'][:100]}...\n"
        
        return summary
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        if not self.conversation_history:
            return {
                'total_invocations': 0,
                'average_response_time': 0
            }
        
        return {
            'total_invocations': len(self.conversation_history),
            'tools_registered': len(self.tools),
            'model_id': self.model_id
        }
    
    def __repr__(self) -> str:
        return f"BedrockAgent(name='{self.name}', model='{self.model_id}')"


# ============================================================================
# Tool Decorator (for Strands-style tool registration)
# ============================================================================

def tool(func):
    """
    Decorator to mark a method as an agent tool.
    
    Usage:
        @tool
        def my_tool(self, param: str) -> dict:
            '''Tool description'''
            return {"result": "value"}
    """
    func._is_tool = True
    return func


# ============================================================================
# Utility Functions
# ============================================================================

def create_agent_prompt(
    task: str,
    context: Dict[str, Any],
    examples: Optional[List[str]] = None
) -> str:
    """
    Create a structured prompt for an agent.
    
    Args:
        task: Task description
        context: Context information
        examples: Optional list of example outputs
    
    Returns:
        Formatted prompt string
    """
    prompt = f"Task: {task}\n\n"
    
    prompt += "Context:\n"
    for key, value in context.items():
        prompt += f"- {key}: {value}\n"
    
    if examples:
        prompt += "\nExamples:\n"
        for i, example in enumerate(examples, 1):
            prompt += f"{i}. {example}\n"
    
    return prompt


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example agent implementation
    class ExampleAgent(BedrockAgent):
        """Example agent for testing"""
        
        def __init__(self):
            super().__init__(
                name="ExampleAgent",
                description="Example agent for demonstration",
                instructions="You are a helpful assistant that provides clear, concise answers."
            )
        
        async def execute(self, input_data: str) -> Dict[str, Any]:
            """Execute example task"""
            response = self.invoke_model(f"Analyze this: {input_data}")
            return {
                'status': 'success',
                'input': input_data,
                'analysis': response
            }
    
    # Test the agent
    import asyncio
    
    async def test():
        agent = ExampleAgent()
        result = await agent.execute("Hello, world!")
        print(json.dumps(result, indent=2))
        
        # Get metrics
        print("\nAgent Metrics:")
        print(json.dumps(agent.get_metrics(), indent=2))
    
    asyncio.run(test())

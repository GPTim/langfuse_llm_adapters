from typing import Any, List, Type
import re, os
from pydantic import ConfigDict
from langchain_core.prompt_values import ChatPromptValue
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from typing import Any, List, Type, Optional
import re
from pydantic import ConfigDict, field_validator, Field, model_validator, BaseModel
from langchain_core.prompt_values import ChatPromptValue

from cat.factory.custom_llm import CustomOllama, CustomOpenAI
from cat.mad_hatter.decorators import hook
from cat.factory.llm import LLMSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat

import google.oauth2.service_account
import google.auth.transport.requests
import json


class ReasoningLLMMixin:
    """Mixin class that adds reasoning processing capabilities."""

    def disable_reasoning_in_prompt(self, input):
        """Process input to handle reasoning/thinking mode."""
        if not self.reasoning:
            # prepending "/no_think" to each inputs in order to avoid reasoning by default
            if isinstance(input, list) and isinstance(input[0], dict):
                input[0]['content'] = f"/no_think\n{input[0]['content']}"
            elif hasattr(input, 'messages') and hasattr(input.messages[0], 'content'):
                input.messages[0].content = "/no_think\n" + input.messages[0].content
            else:
                log.warning("Unable to set /no_think, this might break GPTIM a little bit")
        return input


    def remove_reasoning_section(self, content):
        """Hide reasoning section from content if configured."""
        if self.hide_reasoning_section and isinstance(content, str):
            regex = re.compile(r'<think\b[^>]*>.*?</think>\n?', flags=re.DOTALL)
            return regex.sub('', content)
        return content
    

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        input = self.disable_reasoning_in_prompt(input)
        
        result = super().invoke(input, config, stop=stop, **kwargs)
        
        if hasattr(result, 'content'):
            result.content = self.remove_reasoning_section(result.content)
            
        return result


class CustomOllamaWithLangfuse(ReasoningLLMMixin, CustomOllama):
    """Ollama LLM with Langfuse integration."""
    
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="")
    reasoning: bool = Field(default=False)
    hide_reasoning_section: bool = Field(default=True)
    callbacks: List[Any] = Field(default_factory=list)

    def __init__(self, **kwargs: Any) -> None:

        ollama_params = {
            "num_ctx": kwargs.get('num_ctx'),
            "base_url": kwargs.get("base_url"),
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature")
        }

        CustomOllama.__init__(self, **ollama_params)
        
        # Set Langfuse parameters
        self.langfuse_public_key = kwargs.get('langfuse_public_key', '')
        self.langfuse_secret_key = kwargs.get('langfuse_secret_key', '')
        self.langfuse_host = kwargs.get('langfuse_host', '')
        self.reasoning = kwargs.get('reasoning', False)
        self.hide_reasoning_section = kwargs.get('hide_reasoning_section', True)
        self.callbacks = []


class CustomOpenaiLikeWithLangfuse(ReasoningLLMMixin, CustomOpenAI):
    """OpenAI-like LLM with Langfuse integration."""
    
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="")
    reasoning: bool = Field(default=False)
    hide_reasoning_section: bool = Field(default=True)
    callbacks: List[Any] = Field(default_factory=list)
    
    def __init__(self, **kwargs: Any) -> None:
        if "url" in kwargs and kwargs["url"].endswith("/"):
            kwargs["url"] = kwargs["url"][:-1]
            
        openai_params = {
            "openai_api_key":  kwargs.get("openai_api_key", "meow"),
            "url": kwargs.get("url"),
            "model_name": kwargs.get("model_name"),
            "temperature": kwargs.get("temperature"),
            "streaming": kwargs.get("streaming"),
            "timeout": kwargs.get("timeout"),
            "max_tokens": kwargs.get("max_tokens")
        }

        CustomOpenAI.__init__(self, **openai_params)

        # Set Langfuse parameters
        self.langfuse_public_key = kwargs.get('langfuse_public_key', '')
        self.langfuse_secret_key = kwargs.get('langfuse_secret_key', '')
        self.langfuse_host = kwargs.get('langfuse_host', '')
        self.reasoning = kwargs.get('reasoning', False)
        self.hide_reasoning_section = kwargs.get('hide_reasoning_section', True)
        self.callbacks = []


SERVICE_ACCOUNT_KEY_FILE = "/app/cat/data/tim-cdc-svi-cu00003745p1-l2-c56d1fa6bea7.json"
PROJECT_ID = "tim-cdc-svi-cu00003745p1-l2"  
LOCATION = "global"  
MODEL_NAME = "openai/gpt-oss-120b-maas" #"google/gemini-3-pro-preview"
TEMPERATURE = 1.0

class CustomVertexOpenaiLikeWithLangfuse(CustomOpenaiLikeWithLangfuse):
    """Vertex OpenAI-like LLM with Langfuse integration and OAuth token management."""
    
    service_account_key_json: str = Field(default="")
    project_id: str = Field(default="")
    location: str = Field(default="global")
    
    def __init__(self, **kwargs: Any) -> None:
        # Extract Vertex-specific params
        service_account_key_json = kwargs.get('service_account_key_json', '')
        project_id = kwargs.get('project_id', '')
        location = kwargs.get('location', 'global')
        
        # Generate OAuth token
        token = ""
        if service_account_key_json and project_id:
            try:
                service_account_info = json.loads(service_account_key_json)
                credentials = google.oauth2.service_account.Credentials.from_service_account_info(
                    service_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                credentials.refresh(google.auth.transport.requests.Request())
                token = credentials.token
            except Exception as e:
                log.error(f"Error generating OAuth token: {str(e)}")
                raise ValueError(f"Failed to authenticate with Vertex AI: {str(e)}")
        
        # Build URL and prepare params for parent
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
        
        # Prepare params for parent init (following the same pattern as CustomOpenaiLikeWithLangfuse)
        openai_params = {
            "openai_api_key": token,
            "url": url,
            "model_name": kwargs.get("model_name"),
            "temperature": kwargs.get("temperature"),
            "streaming": kwargs.get("streaming"),
            "timeout": kwargs.get("timeout"),
            "max_tokens": kwargs.get("max_tokens")
        }

        CustomOpenAI.__init__(self, **openai_params)

        # Set additional parameters AFTER parent init (following the pattern)
        self.langfuse_public_key = kwargs.get('langfuse_public_key', '')
        self.langfuse_secret_key = kwargs.get('langfuse_secret_key', '')
        self.langfuse_host = kwargs.get('langfuse_host', '')
        self.reasoning = kwargs.get('reasoning', False)
        self.hide_reasoning_section = kwargs.get('hide_reasoning_section', True)
        self.callbacks = []
        
        # Set Vertex-specific params
        self.service_account_key_json = service_account_key_json
        self.project_id = project_id
        self.location = location
    
    def _get_oauth_token(self) -> str:
        """Generate fresh OAuth token from service account credentials."""
        try:
            service_account_info = json.loads(self.service_account_key_json)
            credentials = google.oauth2.service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(google.auth.transport.requests.Request())
            return credentials.token
        except Exception as e:
            log.error(f"Error generating OAuth token: {str(e)}")
            raise
    
    def invoke(self, input, config=None, *, stop=None, **kwargs):
        """Override invoke to refresh token before each call."""
        try:
            fresh_token = self._get_oauth_token()
            self.openai_api_key = fresh_token
        except Exception as e:
            log.error(f"Failed to refresh OAuth token: {str(e)}")
        
        return super().invoke(input, config, stop=stop, **kwargs)


class LLMOllamaConfigWithLangfuse(LLMSettings):
    base_url: str
    model: str = "llama3"
    num_ctx: int = 2048
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.1
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    reasoning: bool = False
    hide_reasoning_section: bool = True

    _pyclass: Type = CustomOllamaWithLangfuse
    
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Ollama With Langfuse",
            "description": "Configuration for Ollama adapter with Langfuse as observability tool",
        }
    )


class LLMOpenaiLikeConfigWithLangfuse(LLMSettings):
    url: str
    model_name: str = "llama3"
    temperature: float = 0.1
    timeout: float = 3600.0
    max_tokens: int = 32768
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    reasoning: bool = False
    hide_reasoning_section: bool = True
    streaming: bool = True

    _pyclass: Type = CustomOpenaiLikeWithLangfuse
    
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI-like With Langfuse",
            "description": "Configuration for OpenAI-like API scheme adapter with Langfuse as observability tool",
        }
    )


class LLMVertexOpenaiLikeConfigWithLangfuse(LLMSettings):
    """Configuration for Vertex AI OpenAI-like API with Langfuse observability."""
    
    service_account_key_json: str = Field(
        description="Google Cloud service account key as JSON string"
    )
    project_id: str = Field(
        description="Google Cloud project ID"
    )
    location: str = Field(
        default="global",
        description="Google Cloud location for Vertex AI endpoint"
    )
    model_name: str = Field(
        default="openai/gpt-oss-120b-maas",
        description="Model name to use (e.g., openai/gpt-oss-120b-maas, google/gemini-3-pro-preview)"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0
    )
    timeout: float = Field(default=3600.0)
    max_tokens: int = Field(default=32768)
    streaming: bool = Field(default=True)
    
    # Langfuse configuration
    langfuse_host: str = Field(default="")
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    
    # Reasoning configuration
    reasoning: bool = Field(default=False)
    hide_reasoning_section: bool = Field(default=True)

    _pyclass: Type = CustomVertexOpenaiLikeWithLangfuse
    
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Vertex AI OpenAI-like With Langfuse",
            "description": "Configuration for Google Cloud Vertex AI with OpenAI-compatible API and Langfuse observability",
            "link": "https://cloud.google.com/vertex-ai/docs"
        }
    )
    
    @field_validator('service_account_key_json')
    @classmethod
    def validate_json_string(cls, v: str) -> str:
        """Validate that the service account key is valid JSON."""
        if not v:
            raise ValueError("Service account key JSON cannot be empty")
        try:
            json.loads(v)
        except json.JSONDecodeError:
            raise ValueError("Service account key must be valid JSON string")
        return v


@hook
def before_cat_reads_message(user_message_json, cat: StrayCat):
    """Hook to set up Langfuse callbacks before processing user message."""

    if hasattr(cat._llm, "langfuse_public_key"):
        try:
            # Get user session ID (Keycloak session ID) and groups (-> tags)
            user_info = getattr(cat.working_memory.user_message_json, "user", None)
            user_groups = []
            if user_info:
                sid = user_info.get("sid", cat.user_data.id)
                user_groups = user_info.get('gptim', {}).get("groups", [])
            else:
                sid = cat.user_data.id
            # See: https://github.com/orgs/langfuse/discussions/2658
            os.environ["LANGFUSE_HOST"] = cat._llm.langfuse_host
            os.environ["LANGFUSE_PUBLIC_KEY"] = cat._llm.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = cat._llm.langfuse_secret_key
            langfuse = Langfuse()
            trace = langfuse.trace(user_id=cat.user_id, session_id=sid, tags=user_groups)
            langfuse_handler = trace.get_langchain_handler(
                update_parent=True  # add i/o to trace itself as well
            )
            cat._llm.callbacks = [langfuse_handler]
        except Exception as e:
            log.error(f"Error setting up Langfuse callback: {str(e)}")

    return user_message_json


@hook
def before_cat_sends_message(message, cat: StrayCat):
    """Hook to attach Langfuse trace ID to outgoing messages."""
    
    if hasattr(cat._llm, "callbacks") and cat._llm.callbacks:
        for callback in cat._llm.callbacks:
            if isinstance(callback, CallbackHandler) and hasattr(callback, "trace") and hasattr(callback.trace, "id"):
                message.langfuse_trace_id = callback.trace.id
                break
                    
    return message


@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(LLMOllamaConfigWithLangfuse)
    allowed.append(LLMOpenaiLikeConfigWithLangfuse)
    allowed.append(LLMVertexOpenaiLikeConfigWithLangfuse)

    return allowed

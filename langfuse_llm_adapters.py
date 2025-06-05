from typing import Any, List, Type
import re
from pydantic import ConfigDict
from langchain_core.prompt_values import ChatPromptValue
from langfuse.callback import CallbackHandler

from typing import Any, List, Type, Optional
import re
from pydantic import ConfigDict, field_validator, Field, model_validator, BaseModel
from langchain_core.prompt_values import ChatPromptValue
from langfuse.callback import CallbackHandler

from cat.factory.custom_llm import CustomOllama, CustomOpenAI
from cat.mad_hatter.decorators import hook
from cat.factory.llm import LLMSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat




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
            "openai_api_key": "meow",
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




@hook
def before_cat_reads_message(user_message_json, cat: StrayCat):
    """Hook to set up Langfuse callbacks before processing user message."""
    
    if hasattr(cat._llm, "langfuse_public_key"):
        try:
            langfuse_handler = CallbackHandler(
                public_key=cat._llm.langfuse_public_key,
                secret_key=cat._llm.langfuse_secret_key,
                host=cat._llm.langfuse_host,
                user_id=cat.user_id,
                session_id=cat.user_data.id
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
    return allowed

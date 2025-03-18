from typing import Any, List, Type

from langchain_openai import ChatOpenAI
from pydantic import ConfigDict
from cat.mad_hatter.decorators import hook
from cat.factory.llm import LLMSettings
from cat.log import log
from langfuse.callback import CallbackHandler
from cat.looking_glass.stray_cat import StrayCat


class CustomOllamaWithLangfuse(ChatOpenAI):

    langfuse_public_key = None
    langfuse_secret_key = None
    langfuse_host = None

    def __init__(self, **kwargs: Any) -> None:
        if kwargs["base_url"].endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        
        kwargs['model_kwargs'] = {
            "frequency_penalty": kwargs["frequency_penalty"]
        }
        langfuse_public_key=kwargs["langfuse_public_key"]
        langfuse_secret_key=kwargs["langfuse_secret_key"]
        langfuse_host=kwargs["langfuse_host"]
        kwargs.pop('langfuse_public_key', None)
        kwargs.pop('langfuse_secret_key', None)
        kwargs.pop('langfuse_host', None)
        kwargs.pop('frequency_penalty', None)
        super().__init__(api_key='ollama', **kwargs)
        self.langfuse_public_key=langfuse_public_key
        self.langfuse_secret_key=langfuse_secret_key
        self.langfuse_host=langfuse_host



class LLMOllamaConfigWithLangfuse(LLMSettings):
    base_url: str
    model_name: str = "llama3"
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    max_tokens: int = 2048
    n: int = 1
    frequency_penalty: float = 1.1
    temperature: float = 0.8
    streaming: bool = True

    _pyclass: Type = CustomOllamaWithLangfuse
    
    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "LLM OpenAI-Like With Langfuse",
            "description": "Configuration for LLM OpenAI-Like With Langfuse",
        }
    )


@hook  # default priority = 1 
def before_cat_reads_message(user_message_json, cat: StrayCat):
    if isinstance(cat._llm, CustomOllamaWithLangfuse):
        llm: CustomOllamaWithLangfuse = cat._llm
        langfuse_handler = CallbackHandler(
            public_key=llm.langfuse_public_key,
            secret_key=llm.langfuse_secret_key,
            host=llm.langfuse_host,
            user_id = cat.user_id,
            session_id = cat.user_data.id
        )
        llm.callbacks = [langfuse_handler]
    return user_message_json


@hook  # default priority = 1
def before_cat_sends_message(message, cat: StrayCat):
    if isinstance(cat._llm, CustomOllamaWithLangfuse):
        this_llm = cat._llm
        if hasattr(this_llm, "callbacks"):
            for callback in this_llm.callbacks:
                if isinstance(callback, CallbackHandler):
                    trace_id = callback.trace.id
                    message.langfuse_trace_id = trace_id
                    
    return message


@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(LLMOllamaConfigWithLangfuse)
    return allowed



from typing import Any, List, Type

from langchain_openai import ChatOpenAI
from pydantic import ConfigDict
from cat.mad_hatter.decorators import hook
from cat.factory.llm import LLMSettings
from cat.log import log
from langfuse.callback import CallbackHandler
from cat.looking_glass.stray_cat import StrayCat


class CustomOllamaWithLangfuse(ChatOpenAI):

    def __init__(self, **kwargs: Any) -> None:
        if kwargs["base_url"].endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        
        langfuse_handler = CallbackHandler(
            public_key=kwargs["langfuse_public_key"],
            secret_key=kwargs["langfuse_secret_key"],
            host=kwargs["langfuse_host"]
        )

        kwargs['model_kwargs'] = {
            "frequency_penalty": kwargs["frequency_penalty"]
        }
        kwargs.pop('langfuse_public_key', None)
        kwargs.pop('langfuse_secret_key', None)
        kwargs.pop('langfuse_host', None)
        kwargs.pop('frequency_penalty', None)
        super().__init__(api_key='ollama', callbacks=[langfuse_handler], **kwargs)



class LLMOllamaConfigWithLangfuse(LLMSettings):
    base_url: str
    model_name: str = "llama3"
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    max_tokens: int = 2048
    n: int = 64
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
        if hasattr(llm, 'callbacks'):
            for callback in llm.callbacks:
                if isinstance(callback, CallbackHandler):
                    callback.user_id = cat.user_id
                    callback.session_id = cat.user_data.id

    return user_message_json


@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(LLMOllamaConfigWithLangfuse)
    return allowed



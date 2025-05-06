from typing import Any, List, Type
import re
from pydantic import ConfigDict
from langchain_core.prompt_values import ChatPromptValue
from langfuse.callback import CallbackHandler

from cat.factory.custom_llm import CustomOllama
from cat.mad_hatter.decorators import hook
from cat.factory.llm import LLMSettings
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat



class CustomOllamaWithLangfuse(CustomOllama):

    langfuse_public_key = ''
    langfuse_secret_key = ''
    langfuse_host = ''
    reasoning = False
    hide_reasoning_section = True

    def __init__(self, **kwargs: Any) -> None:
        if kwargs["base_url"].endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        
        langfuse_public_key=kwargs["langfuse_public_key"]
        langfuse_secret_key=kwargs["langfuse_secret_key"]
        langfuse_host=kwargs["langfuse_host"]
        reasoning=kwargs["reasoning"]
        hide_reasoning_section=kwargs["hide_reasoning_section"]

        kwargs.pop('langfuse_public_key', None)
        kwargs.pop('langfuse_secret_key', None)
        kwargs.pop('langfuse_host', None)

        super().__init__(api_key='ollama', **kwargs)
        self.langfuse_public_key=langfuse_public_key
        self.langfuse_secret_key=langfuse_secret_key
        self.langfuse_host=langfuse_host
        self.reasoning=reasoning
        self.hide_reasoning_section=hide_reasoning_section


    def invoke(self, input, config = None, *, stop = None, **kwargs):
        # input.messages[0] is HumanMessage type
        if not self.reasoning:
            # prepending "/no_think" to each inputs in order to avoid reasoning by default (this is true for qwen3, I know you are disappointed...)
            # in qwen3 the thinking mode is enabled by default
            if type(input) is list and type(input[0]) is dict:
                input[0]['content'] = f"/no_think\n{input[0]['content']}"
            elif type(input) is ChatPromptValue:
                input.messages[0].content = "/no_think\n" + input.messages[0].content
            else:
                log.warning("Unable to set /no_think, this might break GPTIM a little bit")
                
        
        to_ret = super().invoke(input, config, stop=stop, **kwargs)

        if self.hide_reasoning_section:
            regex = re.compile(r'<think\b[^>]*>.*?</think>\n?', flags=re.DOTALL)
            to_ret.content = regex.sub('', to_ret.content)

        return to_ret



class LLMOllamaConfigWithLangfuse(LLMSettings):
    base_url: str
    model: str = "llama3"
    num_ctx: int = 2048
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.8
    reasoning: bool = False
    hide_reasoning_section: bool = True

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


@hook
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



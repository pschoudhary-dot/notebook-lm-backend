from enum import Enum
from typing import Optional
import os
from dataclasses import dataclass
from openai import OpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


class ModelProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"

@dataclass
class ModelInfo:
    id: str
    provider: ModelProvider
    cost_per_million_tokens: float

class ModelManager:
    def __init__(self):
        self.openai_client = OpenAI(
            api_key="ddc-rCd0jZ1ddZFNkv6qT3ahJkAfZgW43HjRBvu8Qzxuo29Vac4z0V",  # Replace with env variable
            base_url="https://api.sree.shop/v1"
        )
        
        # Define available models
        self.models = {
            "llama-3.3-70b-versatile": ModelInfo("llama-3.3-70b-versatile", ModelProvider.GROQ, 0.3),
            "deepseek-r1-distill-llama-70b": ModelInfo("deepseek-r1-distill-llama-70b", ModelProvider.GROQ, 0.3),
            "claude-3-5-sonnet-20240620": ModelInfo("claude-3-5-sonnet-20240620", ModelProvider.OPENAI, 15),
            "deepseek-r1": ModelInfo("deepseek-r1", ModelProvider.OPENAI, 2.19),
            "deepseek-v3": ModelInfo("deepseek-v3", ModelProvider.OPENAI, 1.28),
            "gpt-4o": ModelInfo("gpt-4o", ModelProvider.OPENAI, 5),
            "Meta-Llama-3.3-70B-Instruct-Turbo": ModelInfo("Meta-Llama-3.3-70B-Instruct-Turbo", ModelProvider.OPENAI, 0.3),
        }
    
    def get_model(self, model_name: str, temperature: float = 0):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model_info = self.models[model_name]
        
        if model_info.provider == ModelProvider.GROQ:
            return ChatGroq(temperature=temperature, model_name=model_name)
        else:   
            return self._create_openai_chain(model_name, temperature)
    
    def _create_openai_chain(self, model_name: str, temperature: float):
        from langchain.chat_models import ChatOpenAI
        
        # Create a wrapper around OpenAI client to make it compatible with LangChain
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key="ddc-rCd0jZ1ddZFNkv6qT3ahJkAfZgW43HjRBvu8Qzxuo29Vac4z0V",  # Replace with env variable
            openai_api_base="https://api.sree.shop/v1"
        )
    
    def model_exists(self, model_name: str) -> bool:
        return model_name in self.models
    
    def list_models(self):
        return [
            {
                "name": name,
                "provider": model.provider.value,
                "cost_per_million_tokens": model.cost_per_million_tokens
            }
            for name, model in self.models.items()
        ]
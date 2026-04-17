import os
import logging
from dotenv import load_dotenv

# 动态导入，避免缺失包时报错
try:
    from langchain_openai.chat_models import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()

class LLMConfig:
    def __init__(self):
        self.llm_provider = 'groq'   # 或 'openrouter'，但 Groq 免费且无需信用卡
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        self.groq_model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        self.openrouter_model = os.getenv('OPENROUTER_MODEL', 'openai/gpt-3.5-turbo')
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_chat_model(self) -> BaseChatModel:
        try:
            if self.llm_provider == 'ollama':
                if ChatOllama is None:
                    raise ImportError("langchain-community is not installed. Please run: poetry add langchain-community")
                self.logger.info(f"Using local Ollama model: {self.ollama_model}")
                return ChatOllama(model=self.ollama_model, temperature=0.2)

            elif self.llm_provider == 'groq':
                if ChatGroq is None:
                    raise ImportError("langchain-groq is not installed. Please run: poetry add langchain-groq")
                groq_api_key = os.getenv('GROQ_API_KEY')
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY not set in .env")
                self.logger.info(f"Using Groq model: {self.groq_model}")
                return ChatGroq(model=self.groq_model, temperature=0.2, api_key=groq_api_key)

            elif self.llm_provider == 'openrouter':
                if ChatOpenAI is None:
                    raise ImportError("langchain-openai is not installed. Please run: poetry add langchain-openai")
                openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
                openrouter_base_url = os.getenv('OPENAI_API_BASE', 'https://openrouter.ai/api/v1')
                if not openrouter_api_key:
                    raise ValueError("OPENROUTER_API_KEY not set in .env")
                self.logger.info(f"Using OpenRouter model: {self.openrouter_model}")
                return ChatOpenAI(
                    model=self.openrouter_model,   # 修正：原为 model=model，现改为 self.openrouter_model
                    temperature=0.2,
                    openai_api_key=openrouter_api_key,
                    openai_api_base=openrouter_base_url,
                    max_retries=1
                )

            else:  # default to OpenAI
                if ChatOpenAI is None:
                    raise ImportError("langchain-openai is not installed. Please run: poetry add langchain-openai")
                self.logger.info(f"Using OpenAI model: {self.openai_model}")
                return ChatOpenAI(
                    model=self.openai_model,
                    temperature=0.2,
                    max_retries=1
                )
        except Exception as e:
            self.logger.error(f"Error initializing chat model: {e}")
            # Fallback to Ollama if available, otherwise raise
            if ChatOllama is not None:
                self.logger.warning("Falling back to Ollama local model")
                return ChatOllama(model=self.ollama_model, temperature=0.2)
            else:
                raise e

    async def generate_text(self, prompt: str) -> str:
        """异步生成文本，供 Agent 生成想法使用"""
        chat_model = self.get_chat_model()
        if hasattr(chat_model, 'ainvoke'):
            response = await chat_model.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            # 同步回退
            response = chat_model.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

    def generate_text_sync(self, prompt: str) -> str:
        """同步生成文本（备用）"""
        chat_model = self.get_chat_model()
        response = chat_model.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def toggle_model(self):
        if self.llm_provider == 'ollama':
            self.llm_provider = 'groq'
        else:
            self.llm_provider = 'ollama'
        self.logger.info(f"Switched to {self.llm_provider} model")

# Global LLM Configuration
llm_config = LLMConfig()
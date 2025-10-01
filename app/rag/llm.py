from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from app.config import OPENROUTER_API_KEY, LLM_MODEL_NAME, OPENROUTER_API_BASE
from langchain_gigachat.chat_models import GigaChat
from app.config import GIGA_KEY

def get_llm(temperature: float = 0.12, ollama: bool = False):
    if ollama == True:
        return ChatOllama(
            model="owl/t-lite:latest",
            #openai_api_key=OPENROUTER_API_KEY,
            #openai_api_base=OPENROUTER_API_BASE,
            base_url="http://localhost:11434",
            temperature=temperature
            )
    else:
        return GigaChat(
            credentials=GIGA_KEY,
            verify_ssl_certs=False,
            temperature=temperature
        )

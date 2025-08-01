from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from app.config import OPENROUTER_API_KEY, LLM_MODEL_NAME, OPENROUTER_API_BASE

def get_llm():
    return ChatOllama(
        model="owl/t-lite:latest",
        #openai_api_key=OPENROUTER_API_KEY,
        #openai_api_base=OPENROUTER_API_BASE,
        base_url="http://localhost:11434",
        temperature=0.15
    )

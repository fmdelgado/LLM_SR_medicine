# tag::llm[]
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
import openai


# Open-source models via Ollama
ollama_llm = Ollama(model="mistral", temperature= 0.0)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# response = ollama_llm.invoke("What can we visit in Hamburg?")


# OpenAI models
openai.api_key = "sk-XXXX"
openai_api_key = openai.api_key
os.environ["OPENAI_API_KEY"] = openai.api_key


openai_llm = ChatOpenAI(openai_api_key=openai.api_key,  model='gpt-3.5-turbo')
openai_embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)



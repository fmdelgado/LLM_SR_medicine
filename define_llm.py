# tag::llm[]
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import XinferenceEmbeddings
from langchain_community.llms.xinference import Xinference
from xinference.client import Client
import os
import openai


# Open-source models via Ollama
ollama_llm = Ollama(model="mistral", temperature= 0.0)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
# response = ollama_llm.invoke("What can we visit in Hamburg?")
# print(response)
#
#

# Open-source models via Xinference server
main_url ="https://llm.cosy.bio"
client = Client(main_url)

models = client.list_models()
model_uids = {model['model_name']: uid for uid, model in models.items()}

xinference_embeddings = XinferenceEmbeddings(server_url=main_url, model_uid=model_uids['gte-large'])

xinference_llm = Xinference(server_url=main_url, model_uid=model_uids['llama-3-instruct'])
# response = xinference_llm.invoke("What can we visit in Hamburg?")
# print(response)




# OpenAI models
openai.api_key = "sk-XXXX"
openai_api_key = openai.api_key
os.environ["OPENAI_API_KEY"] = openai.api_key


openai_llm = ChatOpenAI(openai_api_key=openai.api_key,  model='gpt-3.5-turbo')

openai_embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)



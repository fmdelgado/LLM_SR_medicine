import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import requests, json
from dotenv import load_dotenv
import os
from screener_1 import Screening1

dotenv_path = '/Users/fernando/Documents/Research/LLM_SR_medicine/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/Users/fernando/Documents/Research/academate"
test_df = pd.read_pickle(f'/Users/fernando/Documents/Research/LLM_SR_medicine/data/AI_healthcare/preprocessed_articles_filtered.pkl').head(10)
output_dir = "/Users/fernando/Documents/Research/LLM_SR_medicine/test_newscreener"
criteria = {"presence": "If the article refers to AI, return True. Otherwise, return False",
            "human": "If he article refers to humans return True. Otherwise, return False",
            "chimps": "If The article refers to chimps return True. Otherwise, return False"}

#OPENAI processing
#
# openai_llm = ChatOpenAI(openai_api_key=os.getenv("openai_api"), model='gpt-4o-mini-2024-07-18', temperature=0)
# openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("openai_api"))
# response = openai_llm.invoke("What can we visit in Hamburg?")
# print(response)
#
# #

# #

# #
# test = Screening1(literature_df=test_df, llm=openai_llm, embeddings=openai_embeddings,
#                   criteria_dict=criteria, embeddings_path=f"{output_dir}/test_data/AI_healthcare/embeddings",
#                   vector_store_path=f"{output_dir}/test_data/AI_healthcare", content_column="Record", chunksize=25,
#                   verbose=True)
# results = test.run_screening1()


# CLaude processing
from langchain_community.embeddings import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
import os
import requests
import json

# Authentication details
protocol = "https"
hostname = "chat.cosy.bio"
host = f"{protocol}://{hostname}"
auth_url = f"{host}/api/v1/auths/signin"
api_url = f"{host}/ollama"
account = {
    'email': os.getenv("ollama_user"),
    'password': os.getenv("ollama_pw")
}
auth_response = requests.post(auth_url, json=account)
jwt = auth_response.json()["token"]
headers = {"Authorization": "Bearer " + jwt}



os.environ["ANTHROPIC_API_KEY"] = os.getenv("anthropic_api")
model_name="claude-3-5-sonnet-20240620"
anthropic_llm = ChatAnthropic(model=model_name, temperature=0)
ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text",
                                     headers={"Authorization": "Bearer " + jwt})
response = anthropic_llm.invoke("What can we visit in Hamburg?")
print(response)

#
test = Screening1(literature_df=test_df, llm=anthropic_llm, embeddings=ollama_embeddings,
                  criteria_dict=criteria,  embeddings_path=f"{output_dir}/test_data/AI_healthcare/embeddings_ollama",
                  vector_store_path=f"{output_dir}/test_data/AI_healthcare_{model_name}", content_column="Record",  chunksize=25,
                  verbose=True)
results = test.run_screening1()

# OLLAMA processing
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
import os
import requests
import json

# Authentication details
protocol = "https"
hostname = "chat.cosy.bio"
host = f"{protocol}://{hostname}"
auth_url = f"{host}/api/v1/auths/signin"
api_url = f"{host}/ollama"
account = {
    'email': os.getenv("ollama_user"),
    'password': os.getenv("ollama_pw")
}
auth_response = requests.post(auth_url, json=account)
jwt = auth_response.json()["token"]
headers = {"Authorization": "Bearer " + jwt}

# Initialize your LLM
model_name = "llama3.1:8b"
ollama_llm = ChatOllama(
    base_url="https://chat.cosy.bio/ollama",
    model=model_name,
    temperature=0.0,
    client_kwargs={'headers': headers},
    format="json"
)

# Prepare messages
messages = [
    ("system", "You are a helpful trip advisor."),
    ("human", "What can we visit in Hamburg?"),
    ("assistant", "blablabla"),
    ("human", "How about dark tourism?")
]

# Invoke the LLM
response = ollama_llm.invoke(messages)
print(response)


ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text",
                                     headers={"Authorization": "Bearer " + jwt})

#
test2 = Screening1(literature_df=test_df, llm=ollama_llm, embeddings=ollama_embeddings,
                  criteria_dict=criteria, embeddings_path=f"{output_dir}/test_data/AI_healthcare/embeddings_ollama",
                  vector_store_path=f"{output_dir}/test_data/AI_healthcare_{model_name}", content_column="Record", chunksize=25,
                  verbose=True)
self =test2
results2 = test2.run_screening1()
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
import requests, json
from dotenv import load_dotenv
import os
import sys
sys.path.append("/home/bbb1417/LLM_SR_medicine")
from screener1 import Screening1

# screen -S llm_AI python new_analysis_AI.py

dotenv_path = '/home/bbb1417/LLM_SR_medicine/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/home/bbb1417/LLM_SR_medicine/new_analyses/AI_healthcare"
df = pd.read_pickle(f'/home/bbb1417/LLM_SR_medicine/new_analyses/AI_healthcare/preprocessed_articles_filtered.pkl')
output_dir = "/home/bbb1417/LLM_SR_medicine/new_analyses/AI_healthcare/results"

print(df.columns)

criteria_dict = {
    "AI_functionality_description": "Return true if the study provides a comprehensive description of an AI functionality used in healthcare; otherwise, return false.",

    "Economic_evaluation": " Return true if the study evaluates the economic efficiency and outcomes of an AI application in healthcare, specifically assessing cost-effectiveness or return on investment; otherwise, return false.",

    "Quantitative_healthcare_outcomes": "Return true if the study reports quantitative outcomes in at least one healthcare system, showing measurable impacts such as patient recovery times, treatment efficacy, or cost savings; otherwise, return false.",

    "Relevance_AI_Healthcare": "Return false if the title of the study does not explicitly cover a topic related to AI in healthcare, indicating the study is not primarily focused on AI applications within healthcare; otherwise, return true.",

    "AI_application_description": "Return false if the abstract does not contain a description of an AI application in healthcare, indicating a lack of focus on how AI technologies are implemented or their functional roles within healthcare; otherwise, return true.",

    "Economic_outcome_details": "Return false if the abstract or full text does not elaborate on the quantitative economic outcomes in one healthcare system, failing to provide specific economic data or analysis related to the AI application; otherwise, return true."
}

openai_models = ["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-05-13"]
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("openai_api"))

for openai_model in openai_models:
    print("\n Analyzing with", openai_model)
    outdir = f'{output_dir}/results_{openai_model}'

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created directory {outdir}")
    else:
        print(f"Directory {outdir} already exists.")

    # Set directory permissions to 777
    os.chmod(outdir, 0o777)  # Grant all permissions
    print(f"Set permissions for {outdir}")

    openai_llm = ChatOpenAI(openai_api_key=os.getenv("openai_api"), model=openai_model, temperature=0)

    screening_openai = Screening1(literature_df=df, llm=openai_llm, embeddings=openai_embeddings,
                                  criteria_dict=criteria_dict, embeddings_path=f"{output_dir}/openai_embeddings",
                                  vector_store_path=f"{outdir}", content_column="Record", chunksize=25,
                                  verbose=True)

    results_df = screening_openai.run_screening1()
    results_df.to_pickle(f'{outdir}/results_screening1.pkl')

ollama_models = ['reflection:70b', 'gemma2:27b', 'llama3:8b-instruct-fp16', 'llama3:latest',
                 'gemma:latest', 'mixtral:8x22b', 'mistral:v0.2', 'mistral-nemo:latest', 'meditron:70b',
                 'llama3.1:8b', 'llama3.1:70b', 'gemma2:9b', 'finalend/athene-70b:latest', 'llama3.2:1b',
                 'llama3.2:latest', 'phi3:latest', 'qwen2.5:latest']

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

ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text",
                                     headers={"Authorization": "Bearer " + jwt})

for ollama_model in ollama_models:
    print("\n Analyzing with", ollama_model)
    outdir = f'{output_dir}/results_{ollama_model.replace("/", "_").replace(":", "_")}'

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created directory {outdir}")
    else:
        print(f"Directory {outdir} already exists.")

    # Set directory permissions to 777
    os.chmod(outdir, 0o777)  # Grant all permissions
    print(f"Set permissions for {outdir}")

    try:

        ollama_llm = ChatOllama(base_url="https://chat.cosy.bio/ollama", model=ollama_model, temperature=0.0,
                                client_kwargs={'headers': headers}, format="json")

        screening_ollama = Screening1(literature_df=df, llm=ollama_llm, embeddings=ollama_embeddings,
                                      criteria_dict=criteria_dict, embeddings_path=f"{output_dir}/ollama_embeddings",
                                      vector_store_path=f"{outdir}", content_column="Record", chunksize=25,
                                      verbose=True)
        results_df = screening_ollama.run_screening1()
        results_df.to_pickle(f'{outdir}/results_screening1.pkl')

    except:
        pass

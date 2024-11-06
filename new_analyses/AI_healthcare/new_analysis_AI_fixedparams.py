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
from new_screener1_chroma import Screening1
import pickle

# screen -S llm_AI_chroma_chatcosy_fixedparams python new_analysis_AI_fixedparams.py

dotenv_path = '/home/bbb1417/LLM_SR_medicine/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/home/bbb1417/LLM_SR_medicine/new_analyses/AI_healthcare"
df = pd.read_pickle(f'/home/bbb1417/LLM_SR_medicine/new_analyses/AI_healthcare/preprocessed_articles_filtered.pkl')
output_dir = "/home/bbb1417/LLM_SR_medicine/new_analyses/AI_healthcare/results_chroma_chatcosy_fixedparams"

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

    openai_llm = ChatOpenAI(
        openai_api_key=os.getenv("openai_api"),
        model=openai_model,
        temperature=0.0,  # Match Ollama's temperature setting
        top_p=0.95,  # Match Ollama's top_p
        stop=None,  # Optionally, define stop sequences
        seed=28,
    )

    screening_openai = Screening1(literature_df=df, llm=openai_llm, embeddings=openai_embeddings,
                                  criteria_dict=criteria_dict, embeddings_path=f"{output_dir}/openai_embeddings",
                                  vector_store_path=f"{outdir}", content_column="Record",
                                  verbose=True)

    results_df = screening_openai.run_screening1()
    results_df.to_pickle(f'{outdir}/results_screening1.pkl')

# Read a pickle file
with open("/home/bbb1417/LLM_SR_medicine/new_analyses/ollama_model_list.pkl", 'rb') as f:
    ollama_models = pickle.load(f)
# If you want to see what's in it
print(ollama_models)  # If it's a list/dict it will print directly

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
        ollama_llm = ChatOllama(
            base_url=api_url,
            model=ollama_model,
            temperature=0.0,
            seed=28,
            num_ctx=25000,
            num_predict=-1,
            top_k=100,
            top_p=0.95,
            format="json",
            client_kwargs={'headers': headers})

        screening_ollama = Screening1(literature_df=df, llm=ollama_llm, embeddings=ollama_embeddings,
                                      criteria_dict=criteria_dict, embeddings_path=f"{output_dir}/ollama_embeddings",
                                      vector_store_path=f"{outdir}", content_column="Record",
                                      verbose=True)
        results_df = screening_ollama.run_screening1()
        results_df.to_pickle(f'{outdir}/results_screening1.pkl')

    except:
        pass

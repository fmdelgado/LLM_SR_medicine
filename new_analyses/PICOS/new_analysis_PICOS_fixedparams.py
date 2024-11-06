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

# screen -S llm_picos_chroma_chatcosy python new_analysis_PICOS_fixedparams.py

dotenv_path = '/home/bbb1417/LLM_SR_medicine/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/home/bbb1417/LLM_SR_medicine/new_analyses/PICOS"
df = pd.read_pickle(f'/home/bbb1417/LLM_SR_medicine/new_analyses/PICOS/preprocessed_articles_filtered.pkl')
output_dir = "/home/bbb1417/LLM_SR_medicine/new_analyses/PICOS/results_chroma_chatcosy_fixedparams"

print(df.columns)

criteria_dict = {
    "population": "If the study population comprises patients with musculoskeletal conditions, with no majority having another primary disease or intellectual disabilities, then return true. Otherwise, return false.",
    
    "intervention": "If the treatment involves physiotherapy (techniques like exercises, manual therapy, education, and modalities such as heat, cold, ultrasound, and electrical stimulation to aid in patient recovery, pain reduction, mobility enhancement, and injury prevention), or at least one of the intervention/control group treatments was provided exclusively by physiotherapists, then return true. However, if the treatment of interest was offered by an interdisciplinary team, non-health care professionals, or mostly by a different profession to physiotherapists, then return false.",
    
    "physio_and_other": "In case at least one of the intervention/control group treatments was provided exclusively by physiotherapists, if the intervention includes physiotherapy and another treatment and the other treatment is provided in a comparator group, then return true.",
    
    "e_interventions": "If the study evaluates the economic aspects of E-interventions, digital interventions or eHealth interventions, then return false. Otherwise, return true.",
    
    "control_group": "If there is a control group of any type - for example, wait and see, usual care, placebo, or alternative treatments, then return true. Otherwise, return false.",
    
    "outcome": "If the outcome of the study involves or allows a full economic evaluation, potentially including cost-effectiveness ratios and cost-utility ratios or if the study provides information on the costs and clinical effects of a treatment then return true. Otherwise, return false.",
    
    "study_type": "If the article is not a conference abstract, review, study without results (like a protocol), or model-based study, then return true. Otherwise, return false."
}


openai_models =["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-05-13"]
openai_models =["gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18"]

openai_embeddings = OpenAIEmbeddings(model= "text-embedding-3-large", openai_api_key=os.getenv("openai_api"))

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
        temperature=0.0,           # Match Ollama's temperature setting
        top_p=0.95,                # Match Ollama's top_p
        stop=None,                 # Optionally, define stop sequences
        seed=28,
    )    
    
    screening_openai = Screening1(literature_df=df, llm=openai_llm, embeddings=openai_embeddings,
                      criteria_dict=criteria_dict, embeddings_path=f"{output_dir}/openai_embeddings",
                      vector_store_path=f"{outdir}", content_column="record",
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

ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text", headers={"Authorization": "Bearer " + jwt})

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
                                      vector_store_path=f"{outdir}", content_column="record",
                                      verbose=True)
        results_df = screening_ollama.run_screening1()
        results_df.to_pickle(f'{outdir}/results_screening1.pkl')

    except:
        pass





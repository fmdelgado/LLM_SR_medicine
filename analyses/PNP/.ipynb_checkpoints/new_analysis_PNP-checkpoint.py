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

# screen -S llm_PNP python new_analysis_PNP.py

dotenv_path = '/home/bbb1417/LLM_SR_medicine/.env'
load_dotenv(dotenv_path)

# READING THE DF
workdir = "/home/bbb1417/LLM_SR_medicine/new_analyses/PNP"
df = pd.read_pickle(f'/home/bbb1417/LLM_SR_medicine/new_analyses/PNP/preprocessed_articles_filtered.pkl')
output_dir = "/home/bbb1417/LLM_SR_medicine/new_analyses/PNP/results"

print(df.columns)

criteria_dict = {
    "Disease": "If the study investigates a condition that is classified as a form of hereditary peripheral neuropathy and is recognized based on its genetic basis in medical literature, return True. Otherwise, return False.",
    "Treatment": "If the study focuses on pharmacological therapies or genotype-related dietary changes specifically targeting the neuropathy, return True. If the study primarily investigates non-pharmacological interventions such as physiotherapy, surgery, and genetic counseling, return False.",
    "Human": "If the study involves human participants, return True. If it involves only animal models or in vitro studies, return False.",
    "Genetic": "If human participants in the study are confirmed to have hereditary peripheral neuropathy through direct genetic testing or through family genetic association where further individual testing might not be necessary, return True. This includes confirmation through nucleotide-based assays and clinically accepted diagnostic methods such as specific enzyme activity in biopsies, judged on a case-by-case basis to be as definitive as genetic tests. If the study includes participants diagnosed only based on clinical symptoms without genetic confirmation, return False.",
    "Results": "If results are reported specifically for the genetically confirmed cohort in studies including both genetically confirmed and non-confirmed participants, return True. If the study does not segregate data between genetically confirmed and non-confirmed participants, or if results from credible sources and databases are not exclusively from interventional studies with available results, return False."
}


# openai_models =[ "gpt-4o-mini-2024-07-18", "gpt-4o-2024-05-13"]
# openai_embeddings = OpenAIEmbeddings(model= "text-embedding-3-large", openai_api_key=os.getenv("openai_api"))

# for openai_model in openai_models:
#     print("\n Analyzing with", openai_model)
#     outdir = f'{output_dir}/results_{openai_model}'


#     # Create the directory if it does not exist
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#         print(f"Created directory {outdir}")
#     else:
#         print(f"Directory {outdir} already exists.")

#     # Set directory permissions to 777
#     os.chmod(outdir, 0o777)  # Grant all permissions
#     print(f"Set permissions for {outdir}")

#     openai_llm = ChatOpenAI(openai_api_key=os.getenv("openai_api"), model=openai_model, temperature=0)
    
#     screening_openai = Screening1(literature_df=df, llm=openai_llm, embeddings=openai_embeddings,
#                       criteria_dict=criteria_dict, embeddings_path=f"{output_dir}/openai_embeddings",
#                       vector_store_path=f"{outdir}", content_column="Record", chunksize=25,
#                       verbose=True)
    
#     results_df = screening_openai.run_screening1()
#     results_df.to_pickle(f'{outdir}/results_screening1.pkl')




ollama_models = ['reflection:70b', 'gemma2:27b',
             'llama3:8b-instruct-fp16', 'llama3:latest', 'gemma:latest', 'mixtral:8x22b','mistral:v0.2',
             'phi3:14b', 'mistral-nemo:latest', 'meditron:70b', 'meditron:7b', 'llama3.1:8b',
             'llama3.1:70b', 'gemma2:9b']

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




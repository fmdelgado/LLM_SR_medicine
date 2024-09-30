import pandas as pd
import sys
sys.path.append("/home/bbb1417/academate/app")
from academate import academate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import requests, json
from langchain_community.chat_models import ChatOllama
import os
protocol = "https"
hostname = "chat.cosy.bio"

host = f"{protocol}://{hostname}"

auth_url = f"{host}/api/v1/auths/signin"
api_url = f"{host}/ollama"


# screen -S llm_picos python analyze_PICOS.py

account = {'email': 'Fernando.miguel.delgado-chaves@uni-hamburg.de', 'password': 'FerCosy96!!'}
auth_response = requests.post(auth_url, json=account)
jwt= json.loads(auth_response.text)["token"]


df_dir = "/home/bbb1417/academate/test_data/PICOS/"
df = pd.read_pickle(df_dir+"preprocessed_articles_filtered.pkl")
print(df.head())

df['pdf_path']=f"{df_dir}"+"pdfs/"+df['pdf_name']
embeddings_path=f"{df_dir}/embeddings"
# df = df.head()
print(df.head())


checkpoints_dict = {
    "Population": "If the study population comprises patients with musculoskeletal conditions, with no majority having another primary disease or intellectual disabilities, then return True. Otherwise, return False.",
    "Intervention": "If the treatment involves physiotherapy (techniques like exercises, manual therapy, education, and modalities such as heat, cold, ultrasound, and electrical stimulation to aid in patient recovery, pain reduction, mobility enhancement, and injury prevention), or at least one of the intervention/control group treatments was provided exclusively by physiotherapists, then return True. However, if the treatment of interest was offered by an interdisciplinary team, non-health care professionals, or mostly by a different profession to physiotherapists, then return False. ",
    "Phisiotherapy and another treatment": "In case at least one of the intervention/control group treatments was provided exclusively by physiotherapists, if the intervention includes physiotherapy and another treatment and the other treatment is provided in a comparator group, then return True. ",
    "E-interventions": "If the study evaluates the economic aspects of E-interventions, digital interventions or eHealth interventions, then  return False. Otherwise, return True",
    "Control Group": "If there is a control group of any type - for example, wait and see, usual care, placebo, or alternative treatments, then return True. Otherwise, return False.",
    "Outcome": "If the outcome of the study involves or allows a full economic evaluation, potentially including cost-effectiveness ratios and cost-utility ratios or if the study provides information on the costs and clinical effects of a treatment  then return True. Otherwise, return False.",
    "study type": "If the article is not a conference abstract, review, study without results (like a protocol), or model-based study, then return True. Otherwise, return False.",
}


headers = {"Authorization": f"Bearer {jwt}"}
response = requests.get(f"{api_url}/api/tags", headers=headers)
if response.status_code == 200:
    models = response.json()

# Assuming 'models' is your JSON object loaded from a response
models_data = models['models']
models_below_10b = []

# Loop through each model in the JSON data
for model in models_data:
    # Checking if the model might be a language generative model based on details description
    # This is a very basic heuristic and might need refinement depending on your actual data
    if 'language' in model['details'].get('format', '').lower() or 'llama' in model['details'].get('family', '').lower() or 'bert' in model['details'].get('family', '').lower() or 'gpt' in model['details'].get('family', '').lower():
        # Check if 'parameter_size' exists and if it contains 'B' indicating billion
        if 'parameter_size' in model['details'] and 'B' in model['details']['parameter_size']:
            # Extract the number of parameters as a float and check if it's less than 10 billion
            param_size = float(model['details']['parameter_size'].replace('B', ''))
            if param_size < 10:
                models_below_10b.append(model['name'])

ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text", headers= {"Authorization": "Bearer " + jwt})

# Output the names of the models with less than 10 billion parameters
print("Language generative models with less than 10 billion parameters:")


# models_below_10b = ['llama3.1:8b', 'charlestang06/openbiollm:latest', 'llama3-chatqa:latest', 'llama3-gradient:latest', 'llama3:8b-instruct-fp16', 'llama3:8b-text-fp16', 'llava:latest', 'phi3:latest', 'mistral:v0.2', 'llama3:latest']

models_below_10b = ['llama3:latest',
 'gemma:latest',
 'phi3:latest',
 'mistral:v0.2',
 'phi3:14b',
 'mistral-nemo:latest',
 'medllama2:latest',
 'meditron:70b',
 'meditron:7b',
 'llama3.1:8b',
 'llama3.1:70b',
 'gemma2:9b']

for model_name in models_below_10b:
    print(model_name)
    print("\n Analyzing with", model_name)
    outdir = f'{df_dir}results_{model_name.replace("/", "_")}'


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
        ollama_llm = Ollama(base_url=api_url, model=model_name, temperature=0.0, headers= {"Authorization": "Bearer " + jwt}, format="json")
        ollama_embeddings = OllamaEmbeddings(base_url=api_url, model="nomic-embed-text", headers= {"Authorization": "Bearer " + jwt})
        
        test = academate(topic="AI_healthcare", literature_df=df, llm=ollama_llm,
                         embeddings=ollama_embeddings,
                         criteria_dict=checkpoints_dict,
                         embeddings_path=embeddings_path,
                         vector_store_path=outdir,
                         content_column="record",
                         chunksize=25,
                         verbose=True)
        self = test
        test.run_screening1()
        screening1_df = test.results_screening1
        screening1_df.to_pickle(f'{outdir}/results_screening1.pkl')
        
        df2 = df[df.screening1 == True].copy()
        test = academate(topic="AI_healthcare", literature_df=df2, llm=ollama_llm,
                         embeddings=ollama_embeddings,
                         criteria_dict=checkpoints_dict,
                         embeddings_path=embeddings_path,
                         vector_store_path=outdir,
                         content_column="record",
                         chunksize=25,
                         verbose=True)
        test.run_screening2()
        screening2_df = test.results_screening2
        screening2_df.to_pickle(f'{outdir}/results_screening2.pkl')
        
    except:
        pass
import sys

sys.path.append("/Users/fernando/Documents/Research/aisaac/aisaac/core/")
from screener1_v2 import screening1
import os.path
import pandas as pd

reviewdf = pd.read_pickle("/Users/fernando/Documents/Research/aisaac/data/preprocessed_articles.pkl")
reviewdf = reviewdf.head(20)
test_reviewdf = pd.read_pickle('/Users/fernando/Documents/Research/aisaac/data/test_reviewdf.pkl')
# Reset the index of the subsample DataFrame
checkpoints_dict = {
    "Population": "If the study population comprises patients with musculoskeletal conditions, with no majority having another primary disease or intellectual disabilities, then return True. Otherwise, return False.",
    "Intervention": "If physiotherapists provided one of the intervention/control group treatments alone, then return True. If the treatment of interest was offered by an interdisciplinary team, non-health care professionals, or mostly by a different profession, then return False. If the intervention combines physiotherapy with another treatment and the other treatment is provided in a comparator group, then return True. If the study evaluates the economic aspects of E-interventions, digital interventions or eHealth interventions, then return False",
    "Control Group": "If there is a control group of any type - for example, wait and see, usual care, placebo, or alternative treatments, then return True. Otherwise, return False.",
    "Outcome": "If the outcome of the study involves or allows a full economic evaluation, potentially including cost-effectiveness ratios and cost-utility ratios or if the study provides information on the costs and clinical effects of a treatment, then return True. Otherwise, return False.",
    "study type": "If the article is not a conference abstract, review, study without results (like a protocol), or model-based study, then return True. Otherwise, return False.",
}
test_reviewdf


# Define the new row data
new_data = {
    "record": ["Unfortunately, due to the global nature of the atmosphere,  we cannot have a true control group for studying global warming. It's not possible to isolate a part of Earth and prevent greenhouse gas buildup entirely.  However, scientists rely heavily on historical climate data and computer models to compare Earth's current state to a time before large-scale human influence. This comparison helps isolate the impact of human activity on global temperatures."],
    "database": ["All_incl_articles_20230630.enl"],
    "source-app": ["EndNote"],
    "rec-number": ["9999"],
    "foreign-keys": ["9999"],
    "key": ["9999"],
    "ref-type": ["17"],
    # "contributors": ["Smith, John D.Doyle, Jane E."],
    "titles": ["An innovative approach to global warming"],
    "title": ["An innovative approach to global warming"],
    "related-urls": [None],
    "url": [None],
    "custom6": [None],
    "orig-pub": [None],
    "publisher": [None],
    "secondary-authors": [None],
    "translated-title": [None],
    "concatenated_fields": ["Smith, John D.Doyle, Jane E."],
    "screening1": [False],
    "screening2": [False],
    # Make sure to provide values for all columns here
}

# Convert the dictionary to a DataFrame
new_row = pd.DataFrame(new_data)

# Append the new row to the existing DataFrame
test_reviewdf = pd.concat([new_row, test_reviewdf], ignore_index=True)

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
# Open-source models via Ollama
ollama_llm = Ollama(model="mistral", temperature= 0.0)
ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")


models_dict = {
    'ollama': [ollama_llm, ollama_embeddings],
}
modeltype = 'ollama'
print("analyzing with ", modeltype)
outdir = f'/Users/fernando/Documents/Research/aisaac/data/PICOS_physiotherapy_{modeltype}_testFAKE'

if not os.path.exists(outdir):
    os.makedirs(outdir)
os.chmod(outdir, 0o777)  # Grant all permissions

print(new_row)

test = screening1(literature_df=reviewdf,
                  llm=models_dict[modeltype][0],
                  embeddings=models_dict[modeltype][1],
                  criteria_dict=checkpoints_dict,
                  vector_store_path=outdir,
                  content_column="record")
# self = test
test.run_large_literature()
print(test.results)
test.results.to_pickle(f'{outdir}/results.pkl')

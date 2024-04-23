import os.path
import pickle
import sys

sys.path.append("/Users/fernando/Documents/Research/aisaac/aisaac/utils/")
from define_llm import ollama_llm, ollama_embeddings, xinference_llm, xinference_embeddings, openai_llm, openai_embeddings
from screening1_prompt import *
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import FAISS
from langchain_chroma import Chroma
import chromadb
import os
import faiss
from tqdm import tqdm
import pandas as pd
import time

content_dir = './'

# with open(f"{content_dir}/data/preprocessed_articles.pkl", "rb") as input_file:
#     reviewdf = pickle.load(input_file)
#
# test_reviewdf = reviewdf.groupby('screening1', group_keys=False).apply(lambda x: x.sample(min(len(x), 5)))
# test_reviewdf.to_pickle(f'{content_dir}/data/test_reviewdf.pkl')
test_reviewdf = pd.read_pickle(f'{content_dir}/data/test_reviewdf.pkl')
# Reset the index of the subsample DataFrame
checkpoints_dict = {
    "Population": "If the study population comprises patients with musculoskeletal conditions, with no majority having another primary disease or intellectual disabilities, then return True. Otherwise, return False.",
    "Intervention": "If physiotherapists provided one of the intervention/control group treatments alone, then return True. If the treatment of interest was offered by an interdisciplinary team, non-health care professionals, or mostly by a different profession, then return False. If the intervention combines physiotherapy with another treatment and the other treatment is provided in a comparator group, then return True. If the study evaluates the economic aspects of E-interventions, digital interventions or eHealth interventions, then return False",
    "Control Group": "If there is a control group of any type - for example, wait and see, usual care, placebo, or alternative treatments, then return True. Otherwise, return False.",
    "Outcome": "If the outcome of the study involves or allows a full economic evaluation, potentially including cost-effectiveness ratios and cost-utility ratios or if the study provides information on the costs and clinical effects of a treatment, then return True. Otherwise, return False.",
    "study type": "If the article is not a conference abstract, review, study without results (like a protocol), or model-based study, then return True. Otherwise, return False.",
}



class screening1:
    def __init__(self, literature_df: object, llm: object, embeddings: object, criteria_dict: dict, vector_store_path: str,
                 content_column: str = "content") -> object:
        self.results = None
        self.record2answer = None
        self.missing_records = None
        self.literature_df = literature_df
        self.criteria_dict = criteria_dict
        self.llm = llm
        self.embeddings = embeddings
        self.content_column = content_column
        self.vector_store_path = vector_store_path
        self.db = None

    def docs_to_Chroma(self, docs):
        # docs = loader.load()
        index = Chroma.from_documents(docs, self.embeddings, persist_directory=self.vector_store_path)
        return index

    def load_Chroma(self, path_name):
        index = Chroma(persist_directory=path_name, embedding_function=self.embeddings)
        return index

    def embed_literature_df(self):
        loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)
        # If embedding already present in the vector store, load it
        if os.path.exists(self.vector_store_path):
            self.db = self.load_Chroma(path_name=self.vector_store_path)
            time.sleep(10)
        else:
            self.db = self.docs_to_Chroma(loader.load())

        return self.db

    def screening1(self):

        if os.path.exists(f'{self.vector_store_path}/predicted_criteria.pkl'):
            with open(f'{self.vector_store_path}/predicted_criteria.pkl', 'rb') as file:
                self.record2answer = pickle.load(file)
        else:
            self.record2answer = {}

        if os.path.exists(f'{self.vector_store_path}/missing_records.pkl'):
            with open(f'{self.vector_store_path}/missing_records.pkl', 'rb') as file:
                self.missing_records = pickle.load(file)
        else:
            self.missing_records = set()

        prompt_from_criteria, output_parser = generate_prompt_for_criteria(self.criteria_dict)
        # print(prompt_from_criteria.text)
        chaintype = 'stuff'
        selected_k = 5
        selected_fetch_k = 20

        # recnumber = self.literature_df['record'][0]

        rec_numbers = self.literature_df['record'].to_list()

        for recnumber in tqdm(rec_numbers):
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                   chain_type=chaintype,
                                                   retriever=self.db.as_retriever(search_type="mmr",
                                                                                  search_kwargs={'fetch_k': selected_fetch_k,
                                                                                                 'k': selected_k,
                                                                                                 'filter':{ self.content_column: recnumber}}),
                                                   return_source_documents=False,
                                                   verbose=False)
            for _ in range(5):
                try:
                    if recnumber not in self.record2answer.keys():
                        llm_response = qa_chain.invoke({"query": prompt_from_criteria.text})
                        output = output_parser.parse(llm_response['result'])
                        output = restructure_dict(output, checkpoints_dict)
                        self.record2answer[recnumber] = output
                        self.missing_records.discard(recnumber)
                except:
                    self.missing_records.add(recnumber)

        with open(f'{self.vector_store_path}/predicted_criteria.pkl', 'wb') as file:
            pickle.dump(self.record2answer, file)
        with open(f'{self.vector_store_path}/missing_records.pkl', 'wb') as file:
            pickle.dump(self.missing_records, file)

        print(f"CORRECTLY analyzed {len(self.record2answer)}")
        print(f"INCORRECTLY analyzed {len(self.missing_records)}")

    def structure_output(self):
        data_dict = {}
        for key in self.record2answer.keys():
            data_dict[key] = {}
            for checkpoint_studied in checkpoints_dict.keys():
                data_dict[key][checkpoint_studied] = self.record2answer[key][checkpoint_studied]['label']

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df['record'] = df.index
        df.reset_index(inplace=True, drop=True)
        self.results = self.literature_df.merge(df, on='record', how='left')
        return self.results

    def run(self):
        self.embed_literature_df()
        self.screening1()
        return self.structure_output()



models_dict = {
    'ollama': [ollama_llm, ollama_embeddings],
    'xinference': [xinference_llm, xinference_embeddings],
    'openai': [openai_llm, openai_embeddings]
}

for modeltype in ['ollama', 'xinference', 'openai']:
    print("analyzing with ", modeltype)
    outdir = f'/Users/fernando/Documents/Research/aisaac/data/PICOS_physiotherapy_{modeltype}'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    test = screening1(literature_df = test_reviewdf,
                      llm = models_dict[modeltype][0],
                      embeddings = models_dict[modeltype][1],
                      criteria_dict=checkpoints_dict,
                      vector_store_path=outdir,
                      content_column="record")
# self = test
    test.run()
    test.results.to_pickle(f'{outdir}/results.pkl')
# self.vector_store_path = '/Users/fernando/Documents/Research/aisaac/data/PICOS_physiotherapy'
#
# modeltype = 'ollama'
# outdir = f'/Users/fernando/Documents/Research/aisaac/data/PICOS_physiotherapy_{modeltype}'
# test = screening1(test_reviewdf,
#                   llm =  models_dict[modeltype][0],
#                   embeddings= models_dict[modeltype][1],
#                   criteria_dict=checkpoints_dict,
#                   vector_store_path=outdir,
#                   content_column="record")
# test.run()
#
# self = test
#
# self.embeddings = xinference_embeddings
# self.llm =
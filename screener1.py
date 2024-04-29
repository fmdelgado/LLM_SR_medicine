import os.path
import pickle
import sys
# sys.path.append("/Users/fernando/Documents/Research/aisaac/aisaac/utils/")
# from define_llm import ollama_llm, ollama_embeddings, openai_llm, openai_embeddings
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
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter


class screening1:
    def __init__(self, literature_df: object, llm: object, embeddings: object, criteria_dict: dict,
                 vector_store_path: str,
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
        self.analysis_time = None
        self.db = None

    def docs_to_Chroma(self, docs):
        # docs = loader.load()
        index = Chroma.from_documents(docs, self.embeddings, persist_directory=self.vector_store_path)
        return index

    def load_Chroma(self, path_name):
        index = Chroma(persist_directory=path_name, embedding_function=self.embeddings)
        return index

    def docs_to_FAISS(self, docs, path_name):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=400,
        )
        # docs = loader.load()
        splits = text_splitter.split_documents(docs)
        index = FAISS.from_documents(splits, self.embeddings)
        index.save_local(path_name)
        # self.db = index
        # print(self.db)
        # retrieverdb = self.db.as_retriever(search_type="mmr")
        # print(retrieverdb.invoke("putA?"))
        return index

    def load_FAISS(self, path_name):
        index = FAISS.load_local(path_name, self.embeddings, allow_dangerous_deserialization=True)
        return index

    def embed_literature_df(self, path_name=None):
        if path_name is None:
            path_name = self.vector_store_path
        print("Number of records: ", len(self.literature_df))
        loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)
        # If embedding already present in the vector store, load it
        if os.path.exists(path_name):
            print('loading FAISS DB')
            self.db = self.load_FAISS(path_name=path_name)
            num_documents = len(self.db.index_to_docstore_id)
            print(f"Total number of documents: {num_documents}")
        else:
            t1 = time.time()
            print('Creating new FAISS DB')
            self.db = self.docs_to_FAISS(loader.load(), path_name=path_name)
            print("Time taken to create FAISS DB: ", time.time() - t1)
        time.sleep(10)
        return self.db

    @staticmethod
    def generate_output_parser(checkpoints_dict):
        response_schemas = []

        for i, (checkpoint, description) in enumerate(checkpoints_dict.items(), 1):
            response_schemas.append(
                ResponseSchema(
                    name=f"checkpoint{i}",
                    description=f"True/False, depending on whether {checkpoint} applies to the text.",
                    type='boolean',
                )
            )
            response_schemas.append(
                ResponseSchema(
                    name=f"reason{i}",
                    description=f"The reason for the decision made on {checkpoint}. {description}",
                    type='string',
                )
            )

        # The parser that will look for the LLM output in my schema and return it back to me
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        return output_parser

    @staticmethod
    def instruct_checkpoints(checkpoints_dict):
        key_renaming = {}
        keys_list = list(checkpoints_dict.keys())

        for i in range(len(keys_list)):
            # print(i, keys_list[i])
            key_renaming.update({f"checkpoint{i + 1}": f"checkpoint_{keys_list[i]}"})
            key_renaming.update({f"reason{i + 1}": f"reason_{keys_list[i]}"})

        labels_from_user = "\n%CHECKPOINTS:\n\n" + '\n\n'.join(
            [f"{key} : {value}" for key, value in checkpoints_dict.items()])
        return labels_from_user

    @staticmethod
    def restructure_dict(output, checkpoints_dict):
        key_renaming = {}
        keys_list = list(checkpoints_dict.keys())
        for i in range(len(keys_list)):
            # print(i, keys_list[i])
            key_renaming.update({f"checkpoint{i + 1}": f"checkpoint_{keys_list[i]}"})
            key_renaming.update({f"reason{i + 1}": f"reason_{keys_list[i]}"})

        output = {key_renaming.get(old_key, old_key): value for old_key, value in output.items()}

        new_dict = {}
        for checkpoint in output.keys():
            if 'checkpoint' in checkpoint:
                checkpoint_name = checkpoint.split('_')[1]  # Remove the "checkpoint_" prefix
                new_dict[checkpoint_name] = {
                    'label': output['checkpoint_' + checkpoint_name],
                    'reason': output['reason_' + checkpoint_name]
                }
        return new_dict

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

        chaintype = 'stuff'
        selected_k = 5
        selected_fetch_k = 20

        rec_numbers = self.literature_df['uniqueid'].astype(str).to_list()
        recnumber = rec_numbers[0]
        counter = 0  # Initialize counter

        output_parser = self.generate_output_parser(self.criteria_dict)
        labels_from_user = self.instruct_checkpoints(self.criteria_dict)
        PROMPT_TEMPLATE = """Answer the question based only on the following context:\n{context}\n
            Below are several checkpoints along with their descriptions. After reviewing the provided text, 
            determine if each checkpoint applies to the text. For each checkpoint, state 'Yes' if it applies or 'No'
             if it does not, and briefly explain your reasoning based on the content and context of the text. 
             Only return a JSON object.
             \n
            {question} 
            \n{format_instructions}"""

        PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE,
                                              partial_variables={
                                                  "format_instructions": output_parser.get_format_instructions()})
        init_time = time.time()
        last_three_results = []
        for i, recnumber in enumerate(tqdm(rec_numbers)):
            database_qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=chaintype,
                retriever=self.db.as_retriever(search_type="mmr",
                                               search_kwargs={
                                                   'fetch_k': selected_fetch_k,
                                                   'k': selected_k,
                                                   'filter': {'uniqueid': recnumber}}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False)

            for _ in range(5):
                try:
                    if recnumber not in self.record2answer.keys():
                        llm_response = database_qa.invoke({"query": labels_from_user})
                        if i < 10:  # Only print the results for the first 10 records
                            print(f"Processing record {recnumber}")
                            print(llm_response['result'])
                            print(llm_response['source_documents'])

                        # Check if 'source_documents' is not empty
                        if llm_response['source_documents']:
                            output = output_parser.parse(llm_response['result'])
                            output = self.restructure_dict(output, self.criteria_dict)
                            self.record2answer[recnumber] = output
                            self.missing_records.discard(recnumber)
                        else:
                            print("Retrying...")

                        # Add the new result to the list of the last three results
                        last_three_results.append(llm_response['result'])
                        if len(last_three_results) > 3:
                            last_three_results.pop(0)  # Remove the oldest result if we have more than three

                        # Check if the last three results are all the same
                        if len(last_three_results) == 3 and all(
                                result == last_three_results[0] for result in last_three_results):
                            print(
                                f"Warning: The results for the last three records are all the same: {llm_response['result']}")
                except:
                    print(f"Error processing record {recnumber}")
                    self.missing_records.add(recnumber)

            counter += 1  # Increment counter after each recnumber is processed

            # If counter reaches 20, save results and reset counter
            if counter == 20:
                with open(f'{self.vector_store_path}/predicted_criteria.pkl', 'wb') as file:
                    pickle.dump(self.record2answer, file)
                with open(f'{self.vector_store_path}/missing_records.pkl', 'wb') as file:
                    pickle.dump(self.missing_records, file)
                counter = 0  # Reset counter
        self.analysis_time = time.time() - init_time

    @staticmethod
    def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
    def screening1_v2(self):
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

        selected_k = 5
        selected_fetch_k = 20

        rec_numbers = self.literature_df['uniqueid'].astype(str).to_list()
        recnumber = rec_numbers[0]
        recnumber = '11'

        counter = 0  # Initialize counter

        output_parser = self.generate_output_parser(self.criteria_dict)
        labels_from_user = self.instruct_checkpoints(self.criteria_dict)
        format_instructions = output_parser.get_format_instructions()

        PROMPT_TEMPLATE = """Answer the question based only on the following context:\n{context}\n
            Below are several checkpoints along with their descriptions. After reviewing the provided text, 
            determine if each checkpoint applies to the text. For each checkpoint, state 'Yes' if it applies or 'No'
             if it does not, and briefly explain your reasoning based on the content and context of the text. 
             Only return a JSON object.
             \n{format_instructions}\n{question}\n
            """

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["question", "context"],
            partial_variables={"format_instructions": format_instructions}
        )

        init_time = time.time()
        last_three_results = []


        for i, recnumber in enumerate(tqdm(rec_numbers)):
            retriever = self.db.as_retriever(search_type="mmr",
                                             search_kwargs={
                                                 'fetch_k': selected_fetch_k,
                                                 'k': selected_k,
                                                 'filter': {'uniqueid': recnumber}})

            rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
                    | prompt
                    | self.llm
                    | output_parser
            )

            rag_chain_with_source = RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)

            rag_chain = (
                    {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | output_parser
            )

            for _ in range(5):
                try:
                    if recnumber not in self.record2answer.keys():
                        llm_response = rag_chain_with_source.invoke(labels_from_user)
                        if i < 10:  # Only print the results for the first 10 records
                            print(f"Processing record {recnumber}")
                            print(llm_response['answer'])
                            print(llm_response['context'])

                        # Check if 'source_documents' is not empty
                        if llm_response['context']:
                            output = llm_response['answer']
                            output = self.restructure_dict(output, self.criteria_dict)
                            self.record2answer[recnumber] = output
                            self.missing_records.discard(recnumber)
                        else:
                            print("Retrying...")

                        # Add the new result to the list of the last three results
                        last_three_results.append(llm_response['answer'])
                        if len(last_three_results) > 3:
                            last_three_results.pop(0)  # Remove the oldest result if we have more than three

                        # Check if the last three results are all the same
                        if len(last_three_results) == 3 and all(
                                result == last_three_results[0] for result in last_three_results):
                            print(
                                f"Warning: The results for the last three records are all the same: {llm_response['answer']}")
                except:
                    print(f"Error processing record {recnumber}")
                    self.missing_records.add(recnumber)

            counter += 1  # Increment counter after each recnumber is processed

            # If counter reaches 20, save results and reset counter
            if counter == 20:
                with open(f'{self.vector_store_path}/predicted_criteria.pkl', 'wb') as file:
                    pickle.dump(self.record2answer, file)
                with open(f'{self.vector_store_path}/missing_records.pkl', 'wb') as file:
                    pickle.dump(self.missing_records, file)
                counter = 0  # Reset counter

        with open(f'{self.vector_store_path}/predicted_criteria.pkl', 'wb') as file:
            pickle.dump(self.record2answer, file)
        with open(f'{self.vector_store_path}/missing_records.pkl', 'wb') as file:
            pickle.dump(self.missing_records, file)
        self.analysis_time = time.time() - init_time

    def structure_output(self):
        data_dict = {}
        for key in self.record2answer.keys():
            data_dict[key] = {}
            for checkpoint_studied in self.criteria_dict.keys():
                data_dict[key][checkpoint_studied] = self.record2answer[key][checkpoint_studied]['label']

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df['uniqueid'] = df.index.astype(str)
        df.reset_index(inplace=True, drop=True)
        return df

    def run_large_literature(self):
        self.literature_df['uniqueid'] = self.literature_df.index.astype(str)
        print("Total number of records: ", len(self.literature_df))
        large_literature_df = self.literature_df.copy()
        chunksize = 10

        for start in tqdm(range(0, len(large_literature_df), chunksize)):
            end = start + chunksize
            self.literature_df = large_literature_df[start:end].copy()
            self.embed_literature_df(path_name=f"{self.vector_store_path}/lit_chunk_{start}_{end}")
            self.screening1_v2()
        print(f"CORRECTLY analyzed {len(self.record2answer)}")
        print(f"INCORRECTLY analyzed {len(self.missing_records)}")

        df = self.structure_output()
        self.results = large_literature_df.merge(df, on='uniqueid', how='left')
        self.results['runtime'] = self.analysis_time
        return self.results

    def run(self):
        self.literature_df['uniqueid'] = self.literature_df.index.astype(str)
        self.embed_literature_df()
        self.screening1()
        print(f"CORRECTLY analyzed {len(self.record2answer)}")
        print(f"INCORRECTLY analyzed {len(self.missing_records)}")
        df = self.structure_output()
        self.results = self.literature_df.merge(df, on='uniqueid', how='left')
        self.results['runtime'] = self.analysis_time
        return self.results


# # ## TESTING ------
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# import openai
# content_dir = './'
#
# reviewdf = pd.read_pickle(f"{content_dir}data/preprocessed_articles.pkl")
# reviewdf = reviewdf.head(20)
# test_reviewdf = pd.read_pickle(f'{content_dir}data/test_reviewdf.pkl')
# # Reset the index of the subsample DataFrame
# checkpoints_dict = {
#     "Population": "If the study population comprises patients with musculoskeletal conditions, with no majority having another primary disease or intellectual disabilities, then return True. Otherwise, return False.",
#     "Intervention": "If physiotherapists provided one of the intervention/control group treatments alone, then return True. If the treatment of interest was offered by an interdisciplinary team, non-health care professionals, or mostly by a different profession, then return False. If the intervention combines physiotherapy with another treatment and the other treatment is provided in a comparator group, then return True. If the study evaluates the economic aspects of E-interventions, digital interventions or eHealth interventions, then return False",
#     "Control Group": "If there is a control group of any type - for example, wait and see, usual care, placebo, or alternative treatments, then return True. Otherwise, return False.",
#     "Outcome": "If the outcome of the study involves or allows a full economic evaluation, potentially including cost-effectiveness ratios and cost-utility ratios or if the study provides information on the costs and clinical effects of a treatment, then return True. Otherwise, return False.",
#     "study type": "If the article is not a conference abstract, review, study without results (like a protocol), or model-based study, then return True. Otherwise, return False.",
# }
#
# # print(test_reviewdf.head())
#
# # model_client_url = "http://localhost:11434"
# # ollama_embeddings = OllamaEmbeddings(base_url=model_client_url, model="nomic-embed-text")
#
# # models_list = ["llama2", "qwen", "phi", "zephyr","wizardlm2", "llama3", "mistral", "gemma"]
# # models_list = ["llama2"]
#
#
# # for model in models_list:
# #     ollama_llm = Ollama(base_url=model_client_url, model=model, temperature= 0.0)
#
# #     print("analyzing with ", model)
# #     outdir = f'{content_dir}data/PICOS_physiotherapy_{model}'
#
# #     if not os.path.exists(outdir):
# #         os.makedirs(outdir)
#
# #     os.chmod(outdir, 0o777)  # Grant all permissions
#
# #     test = screening1(literature_df = test_reviewdf,
# #                       llm = ollama_llm,
# #                       embeddings = ollama_embeddings,
# #                       criteria_dict=checkpoints_dict,
# #                       vector_store_path=outdir,
# #                       content_column="record")
# #     # self = test
# #     test.embed_literature_df()
# #     test.screening1()
# #     test.structure_output()
# #     test.results.to_pickle(f'{outdir}/results.pkl')
#
#
# model = 'ollama'
# model_client_url = "http://localhost:11434"
# ollama_llm = Ollama(base_url=model_client_url, model="mistral", temperature=0.0)
# ollama_embeddings = OllamaEmbeddings(base_url=model_client_url, model="nomic-embed-text")
#
#
# openai.api_key = "sk-ssdddd"
# openai_api_key = openai.api_key
# os.environ["OPENAI_API_KEY"] = openai.api_key
#
# openai_llm = ChatOpenAI(openai_api_key=openai.api_key,  model='gpt-3.5-turbo')
# openai_embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
#
#
# models_dict = {
#     'ollama': [ollama_llm, ollama_embeddings],
#     'openai': [openai_llm, openai_embeddings]
# }
#
# print("analyzing with ", model)
# outdir = f'{content_dir}data/PICOS_physiotherapy_{model}'
#
# if not os.path.exists(outdir):
#     os.makedirs(outdir)
# os.chmod(outdir, 0o777)  # Grant all permissions
#
# test = screening1(literature_df=reviewdf,
#                   llm=models_dict[model][0],
#                   embeddings=models_dict[model][1],
#                   criteria_dict=checkpoints_dict,
#                   vector_store_path=outdir,
#                   content_column="record")
# self= test
# self.run_large_literature()
# final_df = self.results
# #
# # self = test
# # self.literature_df['uniqueid'] = self.literature_df.index.astype(str)
# #
# # loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)
# # docs = loader.load()
# # index = Chroma.from_documents(docs, self.embeddings, persist_directory=self.vector_store_path)
# # self.db = index
# # time.sleep(10)
# #
# # print(self.db.get())
# #
# # chaintype = 'stuff'
# # selected_k = 5
# # selected_fetch_k = 20
# # counter = 0  # Initialize counter
# #
# # output_parser = self.generate_output_parser(self.criteria_dict)
# # labels_from_user = self.instruct_checkpoints(self.criteria_dict)
# #
# # PROMPT_TEMPLATE = """Answer the question based only on the following context:\n{context}\n
# # Below are several checkpoints along with their descriptions. After reviewing the provided text,
# # determine if each checkpoint applies to the text. For each checkpoint, state 'Yes' if it applies or 'No'
# #  if it does not, and briefly explain your reasoning based on the content and context of the text.
# #  \n
# # {question}
# # \n{format_instructions}"""
# #
# # PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE,
# #                                       partial_variables={
# #                                           "format_instructions": output_parser.get_format_instructions()})
# #
# # retrieverdb = self.db.as_retriever(search_type="mmr")
# #
# # print("\n TEST INVOKE \n")
# # print(retrieverdb.invoke(
# #     "locomotor function and clinically important reductions in pain. It is recommended that future research investigates methods of increasing compliance with home exercise programmes and evaluates the impact of these interv"))
# #
# # counter = 0  # Initialize counter
# # rec_numbers = self.literature_df['uniqueid'].astype(str).to_list()
# # recnumber = rec_numbers[0]
# #
# # print("\n TEST QA chain \n")
# # print(recnumber, self.literature_df['record'].to_list()[0])
# # print("\n \n")
# #
# # database_qa = RetrievalQA.from_chain_type(
# #     llm=self.llm,  # <1>
# #     chain_type=chaintype,  # <2>
# #     retriever=self.db.as_retriever(search_type="mmr",
# #                                    search_kwargs={
# #                                        'fetch_k': selected_fetch_k,
# #                                        'k': selected_k,
# #                                        'filter': {'uniqueid': recnumber}}),
# #     chain_type_kwargs={"prompt": PROMPT},
# #     return_source_documents=True,
# #     verbose=True)
# #
# # llm_response = database_qa.invoke({"query": labels_from_user})
# # print(llm_response['result'])
# #
# # # test.run()

import os.path
import pickle
import sys
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
import os
from tqdm import tqdm
import pandas as pd
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import io
from langchain.prompts import ChatPromptTemplate
import json
import logging
import requests
from urllib.parse import quote_plus
from typing import Any, TypedDict
import re


class Screening1:
    def __init__(self, llm: object, embeddings: object, criteria_dict: dict, vector_store_path: str,
                 literature_df: pd.DataFrame = None, content_column: str = "record", embeddings_path: str = None,
                 verbose: bool = False, chunksize: int = 25) -> None:

        self.literature_df = literature_df
        self.criteria_dict = criteria_dict
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.content_column = content_column
        self.chunksize = chunksize
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if embeddings_path is None:
            self.embeddings_path = f"{self.vector_store_path}/embeddings"
        else:
            self.embeddings_path = embeddings_path

        self.embeddings_path1 = f"{self.embeddings_path}/screening1_embeddings"
        if not os.path.exists(self.embeddings_path1):
            os.makedirs(self.embeddings_path1)

        self.screening1_dir = f"{self.vector_store_path}/screening1"
        if not os.path.exists(self.screening1_dir):
            os.makedirs(self.screening1_dir)

        self.literature_df.reset_index(drop=True, inplace=True)
        self.literature_df['uniqueid'] = self.literature_df.index.astype(str)

        self.screening1_record2answer = dict()
        self.screening1_missing_records = set()

        self.selected_fetch_k = 20
        self.selected_k = len(self.criteria_dict)

        self.db = None  # Initialize vector store

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
        return index

    def load_FAISS(self, path_name):
        index = FAISS.load_local(path_name, self.embeddings, allow_dangerous_deserialization=True)
        return index

    def embed_literature_df(self, path_name=None):
        if path_name is None:
            path_name = self.embeddings_path1
        if self.verbose:
            print("Number of records: ", len(self.literature_df))
        loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)
        # If embedding already present in the vector store, load it
        if os.path.exists(path_name):
            if self.verbose:
                print('loading FAISS DB')
            self.db = self.load_FAISS(path_name=path_name)
            num_documents = len(self.db.index_to_docstore_id)
            # print(f"Total number of documents: {num_documents}")
        else:
            t1 = time.time()
            if self.verbose:
                print('Creating new FAISS DB')
            self.db = self.docs_to_FAISS(loader.load(), path_name=path_name)
            if self.verbose:
                print("Time taken to create FAISS DB: ", time.time() - t1)
        # time.sleep(10)
        return self.db

    def parse_json_safely(self, json_string):
        try:
            # Extract the JSON object from the response using regex
            json_matches = re.findall(r'\{.*\}', json_string, re.DOTALL)
            if json_matches:
                json_data = json_matches[0]
                return json.loads(json_data)
            else:
                self.logger.error(f"No JSON found in the response: {json_string}")
                return {}  # Return an empty dict if parsing fails
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}\nResponse was: {json_string}")
            return {}

    def prepare_screening_prompt(self):
        formatted_criteria = "\n".join(f"- {key}: {value}" for key, value in self.criteria_dict.items())
        json_structure = json.dumps(
            {key: {"label": "boolean", "reason": "string"} for key in self.criteria_dict.keys()},
            indent=2)

        prompt = ChatPromptTemplate.from_template("""
        Analyze the following scientific article and determine if it meets the specified criteria.
        Only use the information provided in the context.

        Context: {context}

        Criteria:
        {criteria}

        For each criterion, provide a boolean label (true if it meets the criterion, false if it doesn't)
        and a brief reason for your decision.

        **Important**: Respond **only** with a JSON object matching the following structure, and do not include any additional text:

        {json_structure}

        Ensure your response is a valid JSON object.
        """)

        return prompt, formatted_criteria, json_structure

    def prepare_chain(self, retriever, formatted_criteria, json_structure, prompt):
        rag_chain_from_docs = (
                RunnableParallel(
                    {
                        "context": lambda x: "\n\n".join(
                            [doc.page_content for doc in retriever.invoke(x)]),
                        "criteria": lambda x: formatted_criteria,
                        "json_structure": lambda x: json_structure
                    }
                )
                | prompt
                | self.llm
                | (lambda x: self.parse_json_safely(x.content))
        )
        chain = RunnablePassthrough() | rag_chain_from_docs
        return chain

    def process_single_record(self, recnumber, chain):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Invoke the chain with the recnumber as a string
                result = chain.invoke(f"Analyze document {recnumber}")

                if result:
                    return result
                else:
                    self.logger.warning(f"Unexpected response format for record {recnumber}. Retrying...")
            except Exception as e:
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")

            if attempt < max_retries - 1:
                self.logger.info(f"Retrying record {recnumber}... (Attempt {attempt + 2}/{max_retries})")
            else:
                self.logger.error(f"Failed to process record {recnumber} after {max_retries} attempts.")

        return None

    def base_screening(self, screening_type):
        # screening_type = 'screening1'
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        # Load existing screening data
        self.load_existing_screening_data(screening_type)
        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        prompt, formatted_criteria, json_structure = self.prepare_screening_prompt()

        selected_k = len(self.criteria_dict)
        selected_fetch_k = 20
        rec_numbers = self.get_rec_numbers(screening_type)

        init_time = time.time()

        # Create a new list of records to process, excluding those already in record2answer
        records_to_process = [rec for rec in rec_numbers if rec not in record2answer.keys()]
        # recnumber = 0
        for i, recnumber in enumerate(tqdm(records_to_process, desc=f"{screening_type.capitalize()}")):
            try:
                retriever = self.get_retriever(recnumber, screening_type)
                if retriever is None:
                    raise ValueError(f"Failed to create retriever for record {recnumber}")
                chain = self.prepare_chain(retriever, formatted_criteria, json_structure, prompt)
                result = self.process_single_record(recnumber, chain)

                if result is None:
                    missing_records.add(recnumber)
                    self.logger.warning(f"Failed to process record {recnumber} after multiple attempts.")
                else:
                    record2answer[recnumber] = result
                    missing_records.discard(recnumber)  # Remove from missing if it was there

                if i % 20 == 0:
                    self.save_screening_results(screening_type)
            except Exception as e:
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")
                missing_records.add(recnumber)

        # Final check to ensure all records are accounted for
        all_records = set(rec_numbers)
        analyzed_records = set(record2answer.keys())
        missing_records = all_records - analyzed_records

        self.save_screening_results(screening_type)
        setattr(self, f"{screening_type}_analysis_time", time.time() - init_time)

        if self.verbose:
            print(f"Total records: {len(all_records)}")
            print(f"Successfully analyzed: {len(analyzed_records)}")
            print(f"Missing/Failed: {len(missing_records)}")

    def load_existing_screening_data(self, screening_type):
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer_path = f'{screening_dir}/{screening_type}_predicted_criteria.pkl'
        missing_records_path = f'{screening_dir}/{screening_type}_missing_records.pkl'

        if os.path.exists(record2answer_path):
            with open(record2answer_path, 'rb') as file:
                setattr(self, record2answer_attr, pickle.load(file))
        else:
            setattr(self, record2answer_attr, {})

        if os.path.exists(missing_records_path):
            with open(missing_records_path, 'rb') as file:
                setattr(self, missing_records_attr, pickle.load(file))
        else:
            setattr(self, missing_records_attr, set())

        if self.verbose:
            print(f"Loaded {len(getattr(self, record2answer_attr))} existing records for {screening_type}")
            print(f"Loaded {len(getattr(self, missing_records_attr))} missing records for {screening_type}")

    def get_rec_numbers(self, screening_type):
        if screening_type == 'screening1':
            return self.literature_df['uniqueid'].astype(str).to_list()
        else:  # screening2
            return self.results_screening2['uniqueid'].astype(str).to_list()

    def get_retriever(self, recnumber, screening_type):
        if screening_type == 'screening1':
            return self.db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'fetch_k': self.selected_fetch_k,
                    'k': self.selected_k,
                    'filter': {'uniqueid': recnumber}
                }
            )
        else:  # screening2
            pdf_record = self.results_screening2[self.results_screening2['uniqueid'] == recnumber].iloc[0]
            self.embed_article_PDF(pdf_record)
            return self.db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'fetch_k': self.selected_fetch_k,
                    'k': self.selected_k,
                    'filter': {'uniqueid': recnumber}
                }
            )

    def save_screening_results(self, screening_type):
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer = getattr(self, f"{screening_type}_record2answer")
        missing_records = getattr(self, f"{screening_type}_missing_records")

        with open(f'{screening_dir}/{screening_type}_predicted_criteria.pkl', 'wb') as file:
            pickle.dump(record2answer, file)
        with open(f'{screening_dir}/{screening_type}_missing_records.pkl', 'wb') as file:
            pickle.dump(missing_records, file)

    def structure_output(self, answerset: dict = None):
        data_dict = {}
        missing_keys = set()
        for key in answerset.keys():
            data_dict[key] = {}
            for checkpoint_studied in self.criteria_dict.keys():
                if checkpoint_studied in answerset[key]:
                    data_dict[key][checkpoint_studied] = answerset[key][checkpoint_studied].get('label', False)
                else:
                    data_dict[key][checkpoint_studied] = False
                    missing_keys.add(checkpoint_studied)

        if missing_keys and self.verbose:
            print(f"Warning: The following keys were not found in some records: {', '.join(missing_keys)}")

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df['uniqueid'] = df.index.astype(str)
        df.reset_index(inplace=True, drop=True)
        return df, missing_keys

    def select_articles_based_on_criteria(self, df: pd.DataFrame = None):
        if self.verbose:
            print("Selecting articles based on criteria...")
        for column in self.criteria_dict.keys():
            df[column] = df[column].astype(bool)

        columns_needed_as_true = list(self.criteria_dict.keys())
        df['predicted_screening'] = df[columns_needed_as_true].all(axis=1)
        return df

    def run_screening1(self):
        if self.verbose:
            print("Total number of records: ", len(self.literature_df))
        large_literature_df = self.literature_df.copy()

        start_time = time.time()
        for start in tqdm(range(0, len(large_literature_df), self.chunksize), desc="Screening 1"):
            end = start + self.chunksize
            self.literature_df = large_literature_df[start:end].copy()
            db = self.embed_literature_df(path_name=f"{self.embeddings_path1}/lit_chunk_{start}_{end}")
            self.base_screening('screening1')
        self.literature_df = large_literature_df.copy()

        self.analysis_time = time.time() - start_time
        self.results_screening1 = self.post_screening_analysis('screening1')
        self.results_screening1['runtime_scr1'] = self.analysis_time
        # rename criteria columns to add '_scr1'
        self.results_screening1.rename(columns={key: key + '_scr1' for key in self.criteria_dict.keys()}, inplace=True)

        # Add this line to rename 'predicted_screening' to 'predicted_screening1'
        self.results_screening1.rename(columns={'predicted_screening': 'predicted_screening1'}, inplace=True)

        if self.verbose:
            print("Columns in results_screening1:", self.results_screening1.columns)

        return self.results_screening1

    def post_screening_analysis(self, screening_type):
        record2answer = getattr(self, f"{screening_type}_record2answer")
        missing_records = getattr(self, f"{screening_type}_missing_records")

        results_attr = f"results_{screening_type}"
        if hasattr(self, results_attr) and getattr(self, results_attr) is not None:
            total_records = len(getattr(self, results_attr))
        else:
            total_records = len(self.literature_df)  # Fallback to initial dataset

        correctly_analyzed = len(record2answer)
        incorrectly_analyzed = max(0, total_records - correctly_analyzed)  # Ensure this is not negative

        if self.verbose:
            print(f"Total records for {screening_type}: {total_records}")
            print(f"CORRECTLY analyzed: {correctly_analyzed}")
            print(f"INCORRECTLY analyzed: {incorrectly_analyzed}")

        inconsistent_records, extra_keys, missing_keys = self.analyze_model_output(screening_type)

        if inconsistent_records or extra_keys or missing_keys:
            print("There are inconsistencies between your criteria_dict and the model's output.")
            print("Consider updating your criteria_dict or adjusting your model prompt.")

        df, struct_missing_keys = self.structure_output(answerset=record2answer)
        df = self.select_articles_based_on_criteria(df)

        if struct_missing_keys:
            print(f"Warning: The following criteria were missing in some records: {', '.join(struct_missing_keys)}")
            print("This may indicate a mismatch between your criteria_dict and the model's output.")
            print("Consider updating your criteria_dict or adjusting your model prompt.")

        results = self.merge_results(df, screening_type)
        setattr(self, results_attr, results)
        print(f"Columns in {results_attr}:", results.columns)

        return results  # Return the results for further processing if needed

    def merge_results(self, df, screening_type):
        if screening_type == 'screening1':
            return self.literature_df.merge(df, on='uniqueid', how='right')

    def analyze_model_output(self, screening_type):
        all_keys = set()
        inconsistent_records = []
        record2answer = getattr(self, f"{screening_type}_record2answer")

        for record, answers in record2answer.items():
            if isinstance(answers, str):
                try:
                    # Try to parse the string as JSON
                    answers_dict = json.loads(answers)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse answer for record {record} as JSON. Skipping this record.")
                    inconsistent_records.append(record)
                    continue
            elif isinstance(answers, dict):
                answers_dict = answers
            else:
                self.logger.warning(
                    f"Unexpected type for answers of record {record}: {type(answers)}. Skipping this record.")
                inconsistent_records.append(record)
                continue

            record_keys = set(answers_dict.keys())
            all_keys.update(record_keys)

            if record_keys != set(self.criteria_dict.keys()):
                inconsistent_records.append(record)

        extra_keys = all_keys - set(self.criteria_dict.keys())
        missing_keys = set(self.criteria_dict.keys()) - all_keys

        if extra_keys:
            self.logger.warning(f"Extra keys found in model output: {extra_keys}")
        if missing_keys:
            self.logger.warning(f"Keys missing from model output: {missing_keys}")

        return inconsistent_records, extra_keys, missing_keys

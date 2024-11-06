import os
import pickle
import logging
import json
import re
import threading
import tempfile
import shutil
import time
from typing import Any, TypedDict
from tqdm import tqdm
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate


class Screening1:
    def __init__(self, llm: object, embeddings: object, criteria_dict: dict, vector_store_path: str,
                 literature_df: pd.DataFrame = None, content_column: str = "record", embeddings_path: str = None,
                 verbose: bool = False) -> None:

        self.literature_df = literature_df
        self.criteria_dict = criteria_dict
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.content_column = content_column
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Set verbose logging if needed
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)


        if embeddings_path is None:
            self.embeddings_path = os.path.join(self.vector_store_path, "embeddings")
        else:
            self.embeddings_path = embeddings_path

        self.embeddings_path1 = os.path.join(self.embeddings_path, "screening1_embeddings")
        if not os.path.exists(self.embeddings_path1):
            os.makedirs(self.embeddings_path1)

        self.screening1_dir = os.path.join(self.vector_store_path, "screening1")
        if not os.path.exists(self.screening1_dir):
            os.makedirs(self.screening1_dir)

        self.literature_df.reset_index(drop=True, inplace=True)
        self.literature_df['uniqueid'] = self.literature_df.index.astype(str)

        self.screening1_record2answer = dict()
        self.screening1_missing_records = set()

        self.selected_fetch_k = 20
        self.selected_k = len(self.criteria_dict)

        self.db = None  # Initialize vector store

        # Initialize a thread lock for atomic operations
        self.lock = threading.Lock()
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s:%(message)s')

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def atomic_save(self, file_path, data):
        """
        Saves data to a file atomically.
        """
        dir_name = os.path.dirname(file_path)
        with tempfile.NamedTemporaryFile('wb', delete=False, dir=dir_name) as tmp_file:
            pickle.dump(data, tmp_file)
            temp_name = tmp_file.name
        shutil.move(temp_name, file_path)

    def save_screening_results(self, screening_type):
        with self.lock:
            screening_dir = getattr(self, f"{screening_type}_dir")
            record2answer = getattr(self, f"{screening_type}_record2answer")
            missing_records = getattr(self, f"{screening_type}_missing_records")

            # Save atomically
            self.atomic_save(f'{screening_dir}/{screening_type}_predicted_criteria.pkl', record2answer)
            self.atomic_save(f'{screening_dir}/{screening_type}_missing_records.pkl', missing_records)

    def load_existing_screening_data(self, screening_type):
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer_path = f'{screening_dir}/{screening_type}_predicted_criteria.pkl'
        missing_records_path = f'{screening_dir}/{screening_type}_missing_records.pkl'

        # Load predicted criteria
        if os.path.exists(record2answer_path):
            with open(record2answer_path, 'rb') as file:
                predicted_criteria = pickle.load(file)
        else:
            predicted_criteria = {}

        # Load missing records
        if os.path.exists(missing_records_path):
            with open(missing_records_path, 'rb') as file:
                missing_records = pickle.load(file)
        else:
            missing_records = set()

        # Ensure mutual exclusivity
        overlapping_records = set(predicted_criteria.keys()).intersection(missing_records)
        if overlapping_records:
            self.logger.warning(
                f"Found overlapping records in both predicted_criteria and missing_records: {overlapping_records}")
            # Prioritize record2answer and remove from missing_records
            missing_records -= overlapping_records
            self.logger.info(f"Removed overlapping records from missing_records: {overlapping_records}")

        # Set attributes
        setattr(self, record2answer_attr, predicted_criteria)
        setattr(self, missing_records_attr, missing_records)

        if self.verbose:
            print(f"Loaded {len(getattr(self, record2answer_attr))} existing records for {screening_type}")
            print(f"Loaded {len(getattr(self, missing_records_attr))} missing records for {screening_type}")

    def move_record_to_missing(self, record_id, screening_type):
        """
        Move a record from record2answer to missing_records.
        """
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        if record_id in record2answer:
            del record2answer[record_id]
            missing_records.add(record_id)
            self.logger.info(f"Moved record {record_id} from {record2answer_attr} to {missing_records_attr}.")

            # Save changes
            self.save_screening_results(screening_type)
        else:
            self.logger.warning(f"Record {record_id} not found in {record2answer_attr}.")

    def move_record_to_answer(self, record_id, answer, screening_type):
        """
        Move a record from missing_records to record2answer with the provided answer.
        """
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        if record_id in missing_records:
            missing_records.remove(record_id)
            record2answer[record_id] = answer
            self.logger.info(f"Moved record {record_id} from {missing_records_attr} to {record2answer_attr}.")

            # Save changes
            self.save_screening_results(screening_type)
        else:
            self.logger.warning(f"Record {record_id} not found in {missing_records_attr}.")

    def validate_mutual_exclusivity(self, screening_type):
        """
        Validates that no record exists in both record2answer and missing_records.
        """
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        overlapping_records = set(record2answer.keys()).intersection(missing_records)
        if overlapping_records:
            self.logger.error(f"Mutual exclusivity violated! Overlapping records: {overlapping_records}")
            # Resolve overlaps by removing from missing_records
            missing_records -= overlapping_records
            self.logger.info(f"Removed overlapping records from missing_records: {overlapping_records}")
            # Save corrected data
            self.save_screening_results(screening_type)
        else:
            self.logger.info("Mutual exclusivity validated successfully.")

    def run_screening1(self):
        if self.verbose:
            print("Total number of records: ", len(self.literature_df))

        # Load existing screening data and print counts if possible
        self.load_existing_screening_data('screening1')
        record2answer_attr = 'screening1_record2answer'
        missing_records_attr = 'screening1_missing_records'

        print(f"Loaded {len(getattr(self, record2answer_attr))} existing records for screening1")
        print(f"Loaded {len(getattr(self, missing_records_attr))} missing records for screening1")

        start_time = time.time()
        # Embed the entire dataset
        self.embed_literature_df(path_name=self.embeddings_path1)
        # Run screening on the entire dataset
        self.run_screening('screening1')
        self.analysis_time = time.time() - start_time
        self.results_screening1 = self.post_screening_analysis('screening1')
        self.results_screening1['runtime_scr1'] = self.analysis_time
        # Rename criteria columns to add '_scr1'
        self.results_screening1.rename(columns={key: key + '_scr1' for key in self.criteria_dict.keys()}, inplace=True)
        # Rename 'predicted_screening' to 'predicted_screening1'
        self.results_screening1.rename(columns={'predicted_screening': 'predicted_screening1'}, inplace=True)
        if self.verbose:
            print("Columns in results_screening1:", self.results_screening1.columns)
        return self.results_screening1

    def embed_literature_df(self, path_name=None):
        if path_name is None:
            path_name = self.embeddings_path1
        else:
            os.makedirs(path_name, exist_ok=True)
        if self.verbose:
            print("Number of records: ", len(self.literature_df))
        loader = DataFrameLoader(self.literature_df, page_content_column=self.content_column)

        # Load all documents
        all_docs = loader.load()

        # Ensure each document has 'uniqueid' in metadata
        for doc in all_docs:
            if 'uniqueid' not in doc.metadata:
                doc.metadata['uniqueid'] = doc.page_content_hash  # Or generate a unique ID

        # Check if the index files exist
        index_file = os.path.join(path_name, "index.faiss")
        docstore_file = os.path.join(path_name, "index.pkl")

        if os.path.exists(index_file) and os.path.exists(docstore_file):
            if self.verbose:
                print('Loading existing FAISS index...')
            self.db = self.load_FAISS(path_name=path_name)
            indexed_uniqueids = self.get_indexed_uniqueids()
        else:
            if self.verbose:
                print('No existing FAISS index found. Creating a new one...')
            # Create the FAISS index
            self.db = self.docs_to_FAISS(all_docs, path_name)
            indexed_uniqueids = self.get_indexed_uniqueids()
            self.validate_index()
            return self.db  # Since all documents are indexed, we can return here

        # Identify new documents to embed
        new_docs = [doc for doc in all_docs if doc.metadata.get('uniqueid', '') not in indexed_uniqueids]
        if new_docs:
            if self.verbose:
                print(f'Adding {len(new_docs)} new documents to the index...')
            # Ensure 'uniqueid' is in metadata
            for doc in new_docs:
                if 'uniqueid' not in doc.metadata:
                    doc.metadata['uniqueid'] = doc.page_content_hash  # Or generate a unique ID

            # Split and embed new documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=400,
            )
            splits = text_splitter.split_documents(new_docs)

            # Embed in batches with progress bar
            batch_size = 100
            total_batches = len(splits) // batch_size + (1 if len(splits) % batch_size != 0 else 0)

            for i in tqdm(range(total_batches), desc="Embedding new documents"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(splits))
                batch = splits[start_idx:end_idx]
                texts = [doc.page_content for doc in batch]
                try:
                    batch_embeddings = self.embeddings.embed_documents(texts)
                    # Add embeddings and documents to the index
                    self.db.add_embeddings(batch_embeddings, batch)
                except Exception as e:
                    for doc in batch:
                        self.logger.error(f"Error embedding document {doc.metadata.get('uniqueid', 'unknown')}: {e}")

            # Save the updated index
            self.db.save_local(path_name)
        else:
            if self.verbose:
                print('No new documents to embed.')

        self.validate_index()
        return self.db

    def get_indexed_uniqueids(self):
        uniqueids = set()
        for doc_id in self.db.index_to_docstore_id.values():
            doc = self.db.docstore.search(doc_id)
            if doc and 'uniqueid' in doc.metadata:
                uniqueids.add(doc.metadata['uniqueid'])
        return uniqueids

    def validate_index(self):
        indexed_uniqueids = self.get_indexed_uniqueids()
        df_uniqueids = set(self.literature_df['uniqueid'].astype(str))
        missing_uniqueids = df_uniqueids - indexed_uniqueids
        extra_uniqueids = indexed_uniqueids - df_uniqueids

        if missing_uniqueids:
            self.logger.warning(f"Missing {len(missing_uniqueids)} documents in the index.")
            self.logger.debug(f"Missing uniqueids: {missing_uniqueids}")
        else:
            self.logger.info("All documents are correctly embedded in the index.")

        if extra_uniqueids:
            self.logger.warning(f"There are {len(extra_uniqueids)} extra documents in the index not present in the dataframe.")
            self.logger.debug(f"Extra uniqueids: {extra_uniqueids}")

    def run_screening(self, screening_type='screening1'):
        """
        Orchestrates the screening process, ensuring all records are analyzed.
        """
        self.logger.info(f"Starting screening: {screening_type}")

        # Process all records, prioritizing missing_records
        self.base_screening(screening_type=screening_type, reprocess_missing_first=True)

        # After processing, verify that all unique records have been analyzed
        self.generate_status_report(screening_type=screening_type)

        self.logger.info(f"Screening completed: {screening_type}")

    def generate_status_report(self, screening_type):
        """
        Generates a report of the analysis status for all unique records.
        """
        all_unique_ids = set(self.literature_df['uniqueid'].astype(str).tolist())
        record2answer = getattr(self, f"{screening_type}_record2answer")
        missing_records = getattr(self, f"{screening_type}_missing_records")

        analyzed_records = set(record2answer.keys())
        pending_records = set(missing_records)
        not_processed_records = all_unique_ids - analyzed_records - pending_records

        report = {
            'Total Records': len(all_unique_ids),
            'Analyzed Records': len(analyzed_records),
            'Pending Records': len(pending_records),
            'Not Processed Records': len(not_processed_records)
        }

        self.logger.info(f"Status Report for {screening_type}: {report}")

        # Optionally, return the report for further use
        return report

    def base_screening(self, screening_type, reprocess_missing_first=True):
        screening_dir = getattr(self, f"{screening_type}_dir")
        record2answer_attr = f"{screening_type}_record2answer"
        missing_records_attr = f"{screening_type}_missing_records"

        # Load existing screening data
        self.load_existing_screening_data(screening_type)
        record2answer = getattr(self, record2answer_attr)
        missing_records = getattr(self, missing_records_attr)

        # Prepare prompt and other settings
        prompt, formatted_criteria, json_structure = self.prepare_screening_prompt()

        # Determine records to process
        if reprocess_missing_first and missing_records:
            records_to_process = list(missing_records) + [rec for rec in self.get_rec_numbers(screening_type) if
                                                          rec not in record2answer]
        else:
            records_to_process = [rec for rec in self.get_rec_numbers(screening_type) if rec not in record2answer]

        total_records = len(records_to_process)
        self.logger.info(f"Total records to process: {total_records}")

        for i, recnumber in enumerate(tqdm(records_to_process, desc=f"{screening_type.capitalize()}")):
            try:
                retriever = self.get_retriever(recnumber, screening_type)
                if retriever is None:
                    raise ValueError(f"Failed to create retriever for record {recnumber}")
                chain = self.prepare_chain(retriever, formatted_criteria, json_structure, prompt)
                result = self.process_single_record(recnumber, chain)

                if result is None:
                    # Add to missing_records and ensure it's not in record2answer
                    missing_records.add(recnumber)
                    record2answer.pop(recnumber, None)
                    self.logger.warning(f"Failed to process record {recnumber} after multiple attempts.")
                else:
                    # Add to record2answer and ensure it's not in missing_records
                    record2answer[recnumber] = result
                    missing_records.discard(recnumber)

                # Periodically save and validate
                if (i + 1) % 20 == 0 or (i + 1) == total_records:
                    self.save_screening_results(screening_type)
                    self.validate_mutual_exclusivity(screening_type)

            except Exception as e:
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")
                # Add to missing_records and ensure it's not in record2answer
                missing_records.add(recnumber)
                record2answer.pop(recnumber, None)
                self.save_screening_results(screening_type)
                self.validate_mutual_exclusivity(screening_type)

        # Final save and validation
        self.save_screening_results(screening_type)
        self.validate_mutual_exclusivity(screening_type)

    def parse_json_safely(self, json_string):
        try:
            # Attempt to parse the entire content
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                # Extract the JSON object from the response using regex
                json_matches = re.findall(r'\{.*}', json_string, re.DOTALL)
                if json_matches:
                    json_data = json_matches[0]
                    return json.loads(json_data)
                else:
                    self.logger.error(f"No JSON found in the response: {json_string}")
                    return {}  # Return an empty dict if parsing fails
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON after regex extraction: {e}\nResponse was: {json_string}")
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

                if result and isinstance(result, dict):
                    return result
                else:
                    self.logger.warning(f"Unexpected response format for record {recnumber}. Retrying...")
            except Exception as e:
                self.logger.error(f"Error processing record {recnumber}: {str(e)}")
                error_message = str(e).lower()

                if 'rate limit' in error_message or 'too many requests' in error_message:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Rate limit encountered. Sleeping for {sleep_time} seconds.")
                    time.sleep(sleep_time)
                else:
                    # For other types of errors, decide whether to retry or break
                    self.logger.error(f"Non-recoverable error for record {recnumber}.")
                    break  # Break on non-rate limit errors

            if attempt < max_retries - 1:
                self.logger.info(f"Retrying record {recnumber}... (Attempt {attempt + 2}/{max_retries})")
            else:
                self.logger.error(f"Failed to process record {recnumber} after {max_retries} attempts.")

        return None

    def get_rec_numbers(self, screening_type):
        if screening_type == 'screening1':
            return self.literature_df['uniqueid'].astype(str).to_list()
        else:  # screening2
            return self.results_screening2['uniqueid'].astype(str).to_list()

    def get_retriever(self, recnumber, screening_type):
        return self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                'fetch_k': self.selected_fetch_k,
                'k': self.selected_k,
                'filter': {'uniqueid': recnumber}
            }
        )

    def docs_to_FAISS(self, docs, path_name):
        # Ensure each document has metadata including 'uniqueid'
        for doc in docs:
            if 'uniqueid' not in doc.metadata:
                doc.metadata['uniqueid'] = doc.page_content_hash  # Or assign a unique hash if no ID is provided

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=400,
        )
        splits = text_splitter.split_documents(docs)

        if self.verbose:
            print(f"Creating FAISS index with {len(splits)} documents...")

        # Embed documents in batches with progress bar
        batch_size = 100  # Adjust based on your system's capacity
        total_batches = len(splits) // batch_size + (1 if len(splits) % batch_size != 0 else 0)
        embeddings = []
        for i in tqdm(range(total_batches), desc="Embedding documents"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(splits))
            batch = splits[start_idx:end_idx]
            texts = [doc.page_content for doc in batch]
            try:
                batch_embeddings = self.embeddings.embed_documents(texts)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                for doc in batch:
                    self.logger.error(f"Error embedding document {doc.metadata.get('uniqueid', 'unknown')}: {e}")

        # Create the FAISS index with embeddings and metadata
        self.db = FAISS.from_documents(splits, self.embeddings)
        self.db.save_local(path_name)
        return self.db

    def load_FAISS(self, path_name):
        index = FAISS.load_local(path_name, self.embeddings, allow_dangerous_deserialization=True)
        return index

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

    def merge_results(self, df, screening_type):
        if screening_type == 'screening1':
            return self.literature_df.merge(df, on='uniqueid', how='right')
        # Handle other screening types as needed


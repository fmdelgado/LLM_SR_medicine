import pandas as pd
import numpy as np
import pickle

working_directory = "/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/PICOS/"
original_dataset = pd.read_pickle(f"{working_directory}preprocessed_articles.pkl")
unique_original_articles = set(original_dataset['record'].to_list())

no_title_abstract = pd.read_pickle(f"{working_directory}/no_title_abstract.pkl")
original_dataset = original_dataset[~original_dataset['record'].isin(no_title_abstract['record'])]
print(original_dataset.shape)

print(original_dataset.groupby(['screening1']).size(), sum(original_dataset.groupby(['screening1']).size()))
print(original_dataset.groupby(['screening2']).size(), sum(original_dataset.groupby(['screening2']).size()))

unique_original_articles = set(original_dataset.record.astype(str))
record2screening1 = dict(zip(original_dataset.record.astype(str), original_dataset.screening1))
record2screening2 = dict(zip(original_dataset.record.astype(str), original_dataset.screening2))

original_dataset = original_dataset.sort_values(['screening2','concatenated_fields'],ascending=False)
original_dataset.to_pickle(f"{working_directory}preprocessed_articles_filtered.pkl")

model_list = ['gpt-3.5-turbo-0125','llama3:8b', 'mistral:v0.2']
model = 'gpt-3.5-turbo-0125'

print('\n\nMODELS\n\n')
for model in model_list:
    print(model)
    dataset_path = f"{working_directory}embeds_{model}/results.pkl"
    results = pd.read_pickle(dataset_path)
    results['model'] = model
    print(f"Original dataset analyzed by {model} value counts\n", results.screening1.value_counts(), sum(results.screening1.value_counts()))
    results = results[results.record.isin(unique_original_articles)]
    # results['screening1'] = results['record'].map(record2screening1)
    # results['screening2'] = results['record'].map(record2screening2)
    results = results.sort_values(['screening2', 'concatenated_fields'], ascending=False)

    print(f"Analyzed by {model} and filtered by articles with title and abstract\n", results.screening1.value_counts(), sum(results.screening1.value_counts()))
    results_ids = set(results.uniqueid.astype(str))

    with open(f"{working_directory}embeds_{model}/predicted_criteria.pkl", 'rb') as file:
        record2answer = pickle.load(file)
    # Filter record to answers keys that are in the original dataset
    record2answer = {str(k): v for k, v in record2answer.items()}
    record2answer = {k: v for k, v in record2answer.items() if k in results_ids}

    missing_records = set(results.uniqueid.astype(str)) - set(record2answer.keys())


    not_results =  results[results.uniqueid.astype(str).isin(missing_records)]
    results = results[results.uniqueid.astype(str).isin(record2answer.keys())]


    print("analyzed articles", len(results))
    print("missing records", len(not_results))
    print("SUM",len(results) + len(not_results))
    print(f"Actually output articles by {model}\n", results.screening1.value_counts(), sum(results.screening1.value_counts()))

    print('\n From the articles that couldnt be analyzed', not_results.screening1.value_counts(), sum(not_results.screening1.value_counts()))
    print("--------------------")

    results = results.rename(columns={'record': 'Record'})
    results.to_pickle(f"{working_directory}embeds_{model}/results_filtered.pkl")
    with open(f"{working_directory}embeds_{model}/predicted_criteria_filtered.pkl", 'wb') as file:
        pickle.dump(record2answer, file)
    with open(f"{working_directory}embeds_{model}/missing_records_filtered.pkl", 'wb') as file:
        pickle.dump(missing_records, file)

    print('\n')

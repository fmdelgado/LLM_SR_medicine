import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, \
    cohen_kappa_score
import scipy.stats as stats
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.tree import plot_tree

def extract_llm_insights(llm_results, llm_review, llm_record2answer, review_working_directory, llm):
    explanations_df = llm_results[['Record', 'uniqueid'] + llm_review['boolean_column_set']].copy()
    for i, record in explanations_df.iterrows():
        for column in llm_review['columns_needed_as_true']:
            new_column_name = column + "_explanation"
            try:
                explanations_df.at[i, new_column_name] = llm_record2answer[record['uniqueid']][column]['reason']
            except:
                explanations_df.at[i, new_column_name] = 'Missing article'

    explanations_df = explanations_df[['Record'] + llm_review['boolean_column_set']]
    explanations_df.to_excel(f"{review_working_directory}embeds_{llm}/explanations_{llm}.xlsx")


def calculate_pabak(y_true, y_pred):
    # Ensure y_true and y_pred are not empty
    if len(y_true) > 0 and len(y_pred) > 0:
        # Calculate the observed agreement
        agreement_array = np.array(y_true) == np.array(y_pred)
        # Ensure there is at least one True value
        if np.any(agreement_array):
            agreement = np.mean(agreement_array)
            # Calculate PABAK
            pabak = (2 * agreement) - 1
            return pabak
    return np.nan  # Return NaN if conditions are not met


def calculate_odds_ratio(contingency_table):
    # Add a small constant to avoid division by zero
    contingency_table = contingency_table + 0.5
    or_value = (contingency_table[0, 0] / contingency_table[1, 0]) / (contingency_table[0, 1] / contingency_table[1, 1])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return or_value, p_value


def compute_all_metrics(llm_results, selection_approach, llm_results_dict, llm, llm_record2answer, review_original_dataset):
    ratio_of_completion = len(llm_results) / len(review_original_dataset)
    # Compute confusion matrix
    y_true = llm_results['screening1']
    y_pred = llm_results['predicted_screening1']
    cm = confusion_matrix(y_true, y_pred)

    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    print(llm, "\tTP:\t", TP, "\tTN:\t", TN, "\tFP:\t", FP, "\tFN:\t", FN, "TOTAL:\t", len(llm_results))

    # BASIC METRICS
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    pabak = calculate_pabak(y_true, y_pred)

    # ODDS RATIO
    y_true = llm_results['screening2']
    y_pred = llm_results['predicted_screening1']

    contingency_table = np.array([
        [(y_true & y_pred).sum(), (y_true & ~y_pred).sum()],
        [(~y_true & y_pred).sum(), (~y_true & ~y_pred).sum()]
    ])

    or_value, p_value = calculate_odds_ratio(contingency_table)

    # Saving
    llm_results_dict[llm] = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Confusion Matrix": cm,
        "ratio_of_completion": ratio_of_completion,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Matthews correlation coefficient": mcc,
        "Cohen's kappa": kappa,
        "PABAK": pabak,
        "Odds Ratio": or_value,
        "P-value": p_value,
        'ratio_of_completion': ratio_of_completion,
        'succesfully_analyzed_articles': len(llm_record2answer),
        'articles_that_did_not_have_predictions': len(missing_records),
        'selection_approach': selection_approach
    }

    return llm_results_dict


def create_plots_individual_reviews(review_dict, llm_results_dict, llm_results_list, llm_results_df,
                                    review_working_directory):
    # Plot confusion matrices with a general title
    fig, axes = plt.subplots(1, len(review_dict['model_list']), figsize=(8, 3))
    for i, model in enumerate(review_dict['model_list']):
        cm = llm_results_dict[model]['Confusion Matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
        axes[i].set_title(f'{model}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    # Add a general title
    fig.suptitle('Systematic Review', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig(f"{review_working_directory}img/confusion_matrices.png", dpi=300)
    plt.show()

    # Plot performance metrics
    df = llm_results_df[
        ["Precision", "Recall", "F1-score", "Matthews correlation coefficient", "Cohen's kappa", "PABAK"]]
    df.plot(kind='barh', figsize=(10, 7))
    plt.title('Model Performance Metrics')
    plt.xlabel('Score')
    plt.ylabel('Metrics')
    plt.legend(title='Models', bbox_to_anchor=(0.5, -0.1), loc='upper center')
    plt.tight_layout()
    plt.savefig(f"{review_working_directory}img/benchmark.png", dpi=300)
    plt.show()

    # Initialize an empty dictionary to store the PABAK scores
    pabak_scores = {}

    # Add 'screening1' to the model list
    model_list_with_gt = review_dict['model_list'] + ['human']

    # For each model in the model list
    for model1 in model_list_with_gt:
        pabak_scores[model1] = {}
        for model2 in model_list_with_gt:
            y_pred1 = llm_results_list[model1]
            y_pred2 = llm_results_list[model2]
            common_indices = y_pred1.index.intersection(y_pred2.index)
            y_pred1 = y_pred1.loc[common_indices]
            y_pred2 = y_pred2.loc[common_indices]
            pabak_score = calculate_pabak(y_pred1, y_pred2)
            pabak_scores[model1][model2] = pabak_score

    pabak_df = pd.DataFrame(pabak_scores)
    mask = np.tril(np.ones(pabak_df.shape)).astype(bool)
    lower_triangular_df = pabak_df.where(mask)

    plt.figure(figsize=(7, 6))
    sns.heatmap(lower_triangular_df, annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1, center=0)
    plt.title('PABAK Scores (cross-rater agreement)')
    plt.xlabel('Rater')
    plt.ylabel('Rater')
    plt.savefig(f"{review_working_directory}img/crossrater_pabak.png", dpi=300)
    plt.show()

    # Initialize an empty dictionary to store the kappa scores
    kappa_scores = {}

    # For each model in the model list
    for model1 in model_list_with_gt:
        kappa_scores[model1] = {}
        for model2 in model_list_with_gt:
            y_pred1 = llm_results_list[model1]
            y_pred2 = llm_results_list[model2]
            common_indices = y_pred1.index.intersection(y_pred2.index)
            y_pred1 = y_pred1.loc[common_indices]
            y_pred2 = y_pred2.loc[common_indices]
            kappa_score = cohen_kappa_score(y_pred1, y_pred2)
            kappa_scores[model1][model2] = kappa_score

    kappa_df = pd.DataFrame(kappa_scores)
    mask = np.tril(np.ones(kappa_df.shape)).astype(bool)
    lower_triangular_df = kappa_df.where(mask)

    plt.figure(figsize=(7, 6))
    sns.heatmap(lower_triangular_df, annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1, center=0)
    plt.title('Kappa Scores (cross-rater agreement)')
    plt.xlabel('Rater')
    plt.ylabel('Rater')
    plt.savefig(f"{review_working_directory}img/crossrater_kappa.png", dpi=300)
    plt.show()


def compute_per_column_performance(review_dict, llm_results, llm, llm_record2answer, review_original_dataset):
    # Initialize dictionary to store metrics for each column and model
    model_metrics_percolumn = {}
    # Assuming 'results' is a DataFrame containing the model outputs and 'review' is a dictionary
    # with keys including 'columns_needed_as_true'
    for column in review_dict['columns_needed_as_true']:
        # Compare the model's predictions to true values
        llm_results['predicted_screening1'] = llm_results[column] == True
        y_true = llm_results['screening1']  # Assuming 'screening1' contains the true labels
        y_pred = llm_results['predicted_screening1']

        print(column)  # This will print the current column being processed

        if len(y_true) > 0:  # Check if there are enough samples
            # Compute metrics for the current column and store them under that column's key
            temp_dict = compute_all_metrics(llm_results=llm_results,
                                            selection_approach='individual_col',
                                            llm_results_dict={},
                                            llm=llm,
                                            llm_record2answer=llm_record2answer,
                                            review_original_dataset=review_original_dataset)[llm]
            temp_dict['model'] = llm
            model_metrics_percolumn[column] = temp_dict
        else:
            # If no data is available, assign None or an appropriate value
            model_metrics_percolumn[column] = None

    metrics_per_model = pd.DataFrame.from_dict(model_metrics_percolumn, orient='index')
    return metrics_per_model


def plot_CM_per_column(llm_per_column_performance_df, llm_review, review_working_directory):
    # Plot confusion matrices for each column in columns_needed_as_true
    for crit_column in llm_review['columns_needed_as_true']:
        fig, axes = plt.subplots(1, len(llm_review['model_list']), figsize=(12, 5))
        fig.suptitle(f'Confusion Matrices for {crit_column}', fontsize=16)

        for i, llm_model in enumerate(llm_review['model_list']):
            cm = per_column_performance_df[(llm_per_column_performance_df['model']==llm_model) & (llm_per_column_performance_df['index'] == crit_column)]['Confusion Matrix'].iloc[0]
            if cm is not None:
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
                axes[i].set_title(f'{llm_model}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')
            else:
                axes[i].set_title(f'{llm_model}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')
                axes[i].text(0.5, 0.5, 'Not enough data', ha='center', va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{review_working_directory}img/confusion_matrix_{crit_column}.png", dpi=300)
        plt.show()


def split_long_names(name, max_length=20):
    if len(name) > max_length:
        words = name.split()
        split_name = ''
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > max_length:
                split_name += '\n' + word
                current_length = len(word)
            else:
                if split_name:
                    split_name += ' ' + word
                else:
                    split_name = word
                current_length += len(word) + 1
        return split_name
    else:
        return name


def plot_models_corrmatrices(llm_review, review_corr_matrices, review_working_directory):

    # Adjusting the index and column names
    for llm in llm_review['model_list']:
        review_corr_matrices[llm].index = [split_long_names(name) for name in review_corr_matrices[llm].index]
        review_corr_matrices[llm].columns = [split_long_names(name) for name in review_corr_matrices[llm].columns]

    # Plotting the lower triangular matrices
    fig, axes = plt.subplots(1, len(llm_review['model_list']), figsize=(12 + (len(llm_review['model_list']) - 1) * 2, 6))
    for i, llm in enumerate(llm_review['model_list']):
        # Masking the upper triangle
        mask = np.triu(np.ones_like(review_corr_matrices[llm], dtype=bool))

        sns.heatmap(review_corr_matrices[llm].round(2), annot=True, cmap='coolwarm', center=0, ax=axes[i],
                    cbar=(i == len(llm_review['model_list']) - 1), vmin=-1, vmax=1, linecolor='white', linewidths=1, mask=mask)
        axes[i].set_title(f'{llm}')

        # Bold the 'screening1' index
        yticklabels = [label.get_text() for label in axes[i].get_yticklabels()]
        yticklabels = [f'$\\bf{{{label}}}$' if label == 'screening1' else label for label in yticklabels]
        axes[i].set_yticklabels(yticklabels, rotation=0, fontsize=9)  # Set rotation to 0 for y-axis labels
        xticklabels = [label.get_text() for label in axes[i].get_xticklabels()]
        xticklabels = [f'$\\bf{{{label}}}$' if label == 'screening1' else label for label in xticklabels]
        axes[i].set_xticklabels(xticklabels, rotation=90, fontsize=9)  # Set rotation to 90 for x-axis labels
    fig.suptitle(f"Correlation matrices for LLM-predicted criteria in Review {llm_review['name']}", fontsize=11)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{review_working_directory}img/RF/combined_correlation_matrices.png", dpi=300)
    plt.show()


def plot_tree_example(llm_rf_classifier, llm_review, llm_model, review_working_directory, my_X):
    # Get the feature importances
    feature_importances = llm_rf_classifier.feature_importances_
    # Create a pandas series with feature importances
    feature_importances_series = pd.Series(feature_importances, index=my_X.columns)
    # Sort the series in descending order
    sorted_feature_importances = feature_importances_series.sort_values(ascending=False)
    feature_importances_dict[llm_model] = sorted_feature_importances

    # Plot the sorted feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_feature_importances, y=sorted_feature_importances.index)

    # Add labels and title
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f"Review {llm_review['name']} - {llm_model} RF Feature Importances")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
    plt.savefig(f"{review_working_directory}img/RF/featureImpotance_{llm_model}.png", dpi=300)
    plt.show()

    # Plot a tree from the Random Forest
    plt.figure(figsize=(5, 5))
    plot_tree(llm_rf_classifier.estimators_[0], filled=True, feature_names=my_X.columns)
    plt.savefig(f"{review_working_directory}img/RF/tree_{llm_model}.png", dpi=300)
    plt.show()


def plot_RF_CM(review_results_dict_RF, llm_model, llm_review, review_working_directory):
    # Plot the confusion matrix
    plt.figure(figsize=(3, 3))
    sns.heatmap(review_results_dict_RF[llm_model]['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Review {llm_review['name']} - {llm_model}")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Return the performance metrics
    performance_metrics_table = {
        "Precision": np.round(review_results_dict_RF[llm_model]['Precision'], 4),
        "Recall": np.round(review_results_dict_RF[llm_model]['Recall'], 4),
        "F1-score": np.round(review_results_dict_RF[llm_model]["F1-score"], 4),
        "MCC": np.round(review_results_dict_RF[llm_model]['Matthews correlation coefficient'], 4)
    }

    # Create a DataFrame from the performance metrics
    tabledata = pd.DataFrame(performance_metrics_table, index=[0])

    # Create a table with the performance metrics and add it to the plot
    mytable = plt.table(cellText=tabledata.values, colLabels=tabledata.columns, cellLoc='center', loc='bottom',
                      bbox=[0, -0.5, 1, 0.2])

    # Adjust layout to make room for the table
    plt.subplots_adjust(left=0.2, bottom=0.4)
    plt.savefig(f"{review_working_directory}img/RF/RF_CM_{llm_model}.png", dpi=300)
    plt.show()


rw_1_workdir = {'directory': "/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/PICOS/",
                "boolean_column_set": ['screening1', 'screening2', 'Population', 'Intervention',
                                       'Phisiotherapy and another treatment', 'E-interventions', 'Control Group',
                                       'Outcome', 'study type'],
                "columns_needed_as_true": ['Population', 'Intervention', 'Phisiotherapy and another treatment',
                                           'E-interventions', 'Control Group', 'Outcome', 'study type'],
                "model_list": ['gpt-3.5-turbo-0125', 'llama3:8b', 'mistral:v0.2'],
                "name": "I"}

rw_2_workdir = {'directory': "/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/PNP/",
                "boolean_column_set": ['screening1', 'screening2', 'Disease', 'Treatment', 'Human', 'Genetic',
                                       'Results'],
                "columns_needed_as_true": ['Disease', 'Treatment', 'Human', 'Genetic', 'Results'],
                "model_list": ['gpt-3.5-turbo-0125', 'llama3:8b', 'mistral:v0.2'],
                "name": "II"}

rw_3_workdir = {'directory': "/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/AI_healthcare/",
                "boolean_column_set": ['screening1', 'screening2', 'AI Functionality Description',
                                       'Economic Evaluation', 'Quantitative Healthcare Outcomes',
                                       'Relevance to AI in Healthcare', 'AI Application Description',
                                       'Economic Outcome Details'],
                "columns_needed_as_true": ['AI Functionality Description', 'Economic Evaluation',
                                           'Quantitative Healthcare Outcomes', 'Relevance to AI in Healthcare',
                                           'AI Application Description', 'Economic Outcome Details'],
                "model_list": ['gpt-3.5-turbo-0125', 'llama3:8b', 'mistral:v0.2'],
                "name": "III"}

review = rw_1_workdir

all_results_list_alltrue = []
all_results_list_RF = []

for review in [rw_1_workdir, rw_2_workdir, rw_3_workdir]:
    working_directory = review['directory']
    original_dataset = pd.read_pickle(f"{working_directory}preprocessed_articles_filtered.pkl")

    results_dict_alltrue = {}
    results_list_alltrue = {}
    results_dict_RF = {}
    results_list_RF = {}
    per_column_performance = []
    corr_matrices = {}
    feature_importances_dict = {}

    model = 'gpt-3.5-turbo-0125'

    for model in review['model_list']:
        dataset_path = f"{working_directory}embeds_{model}/results_filtered.pkl"
        results = pd.read_pickle(dataset_path)
        # results['model'] = model
        # results.screening1.value_counts()

        with open(f"{working_directory}embeds_{model}/predicted_criteria_filtered.pkl", 'rb') as file:
            record2answer = pickle.load(file)
        with open(f"{working_directory}embeds_{model}/missing_records_filtered.pkl", 'rb') as file:
            missing_records = pickle.load(file)
        for column in review['boolean_column_set']:
            results[column] = results[column].astype(bool)

        # PUT ALL RESULTS TO READABLE FORMAT
        extract_llm_insights(llm_results=results, llm_review=review, llm_record2answer=record2answer,
                             review_working_directory=working_directory, llm=model)

        # EVALUATE THE PREDICTIVE POWER OVER THE SCREENING 1 CLASS OF THE LLM-INFERRED CRITERIA

        per_column_performance.append(compute_per_column_performance(review_dict=review, llm_results=results, llm=model,
                                                                     llm_record2answer=record2answer,
                                                                     review_original_dataset=original_dataset))



        # Compute the predicted screening1 column based on the columns needed as true
        results['predicted_screening1'] = results[review['columns_needed_as_true']].all(axis=1)
        results['predicted_screening1'].value_counts()

        results_list_alltrue[model] = results.predicted_screening1

        results_dict_alltrue.update(compute_all_metrics(llm_results=results, selection_approach="all_true",
                                                        llm_results_dict=results_dict_alltrue,
                                                        llm=model, llm_record2answer=record2answer,
                                                        review_original_dataset=original_dataset))

        rf_input = results[review['boolean_column_set']]
        X = rf_input.drop(['screening1', 'screening2'], axis=1)  # Features
        y = rf_input['screening1']  # Target variable

        # Add the target variable to the features DataFrame for correlation calculation
        df_for_corr = X.copy()
        df_for_corr['screening1'] = y

        # Calculate the correlation matrix
        corr_matrix = df_for_corr.corr()
        corr_matrices[model] = corr_matrix
        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f"Review {review['name']} - {model} Correlation Matrix")
        plt.savefig(f"{working_directory}img/RF/correlationMatrix_{model}.png", dpi=300)
        plt.show()

        # Compute Mutual Information
        mi_scores = mutual_info_classif(X, y, discrete_features=True)
        # Create a DataFrame for Mutual Information scores
        mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
        # Visualize the Mutual Information scores
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Mutual Information', y='Feature',
                    data=mi_df.sort_values(by='Mutual Information', ascending=False))
        plt.title(f"Review {review['name']} - {model} Mutual Information Scores for Each Feature")
        plt.savefig(f"{working_directory}img/RF/MI_{model}.png", dpi=300)
        plt.show()
        # Initialize the Random Forest Classifier with class_weight='balanced'
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        # Perform cross-validation predictions
        y_pred = cross_val_predict(rf_classifier, X, y, cv=5)
        results_list_alltrue[model] = pd.Series(y_pred)

        results['predicted_screening1'] = y_pred
        results_dict_RF.update(compute_all_metrics(llm_results=results, selection_approach="RF",
                                                llm_results_dict=results_dict_RF,
                                                llm=model, llm_record2answer=record2answer, review_original_dataset=original_dataset))

        # Fit the model on the entire dataset
        rf_classifier.fit(X, y)
        plot_tree_example(llm_rf_classifier=rf_classifier, llm_review=review, llm_model=model, review_working_directory=working_directory, my_X=X)
        plot_RF_CM(review_results_dict_RF=results_dict_RF, llm_model=model, llm_review=review, review_working_directory=working_directory)

    plot_models_corrmatrices(llm_review=review, review_corr_matrices=corr_matrices, review_working_directory=working_directory)
    per_column_performance_df = pd.concat(per_column_performance)
    per_column_performance_df.sort_values("Matthews correlation coefficient", ascending=False, inplace=True)
    per_column_performance_df.to_excel(f"{working_directory}img/metrics_per_column.xlsx")
    per_column_performance_df.reset_index(inplace=True)
    plot_CM_per_column(llm_per_column_performance_df=per_column_performance_df, llm_review=review, review_working_directory=working_directory)


    results_list_alltrue['human'] = original_dataset['screening1']
    # Convert the results dictionary to a pandas DataFrame
    results_df = pd.DataFrame(results_dict_alltrue).T
    results_df.to_pickle(f"{working_directory}benchmark_PICOS.pkl")
    create_plots_individual_reviews(review_dict=review, llm_results_dict=results_dict_alltrue, llm_results_list=results_list_alltrue,
                                    llm_results_df=results_df, review_working_directory=working_directory)



    # Transforming the dictionary into a DataFrame
    df = pd.DataFrame(results_dict_alltrue).T.reset_index()
    df = df.rename(columns={'index': 'Model'})
    df['review'] = review['name']
    df.to_pickle(f"{working_directory}results_alltrue.pkl")
    all_results_list_alltrue.append(df)

    # Transforming the dictionary into a DataFrame
    df = pd.DataFrame(results_dict_RF).T.reset_index()
    df = df.rename(columns={'index': 'Model'})
    df['review'] = review['name']
    df.to_pickle(f"{working_directory}results_RF.pkl")
    all_results_list_RF.append(df)


all_results_list_alltrue = pd.concat(all_results_list_alltrue)
all_results_list_RF = pd.concat(all_results_list_RF)
all_results_list_alltrue.to_pickle("/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/all_results_list_alltrue.pkl")
all_results_list_RF.to_pickle("/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/all_results_list_RF.pkl")
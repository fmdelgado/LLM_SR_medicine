import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import ast

workdir = '/Users/fernando/Documents/Research/LLM_SR_medicine/llm_results/'
approach_trues = pd.read_pickle(f"{workdir}all_results_list_alltrue.pkl")
approach_RF = pd.read_pickle(f"{workdir}all_results_list_RF.pkl")

# Concatenate the data from the three reviews
df = pd.concat([approach_trues, approach_RF], ignore_index=True)
df.to_excel(f"{workdir}joined_analysis_withRF.xlsx", index=False)

# Define the metrics and their ranges
metrics = ['Precision', 'Recall', 'F1-score', 'Matthews correlation coefficient', "Cohen's kappa", 'PABAK']
range_0_1 = ['Precision', 'Recall', 'F1-score']
range_neg1_1 = ['Matthews correlation coefficient', "Cohen's kappa", 'PABAK']

df.rename(columns={'selection_approach': 'selection_method'}, inplace=True)

# Update the dataset to include the correct 'All True' and 'RF' labels
df['selection_method'] = df['selection_method'].replace('all_true', '(All True)')
df['selection_method'] = df['selection_method'].replace('RF', '(RF)')

# Replacing only the standalone occurrences of 'I', 'II', and 'III'
df['review'] = df['review'].str.replace(r'\bI\b', 'Review I', regex=True)
df['review'] = df['review'].str.replace(r'\bII\b', 'Review II', regex=True)
df['review'] = df['review'].str.replace(r'\bIII\b', 'Review III', regex=True)

# Create a new column to combine review and selection method for coloring
df['review_selection'] = df['review'] + ' ' + df['selection_method']

# Define the palette for the reviews with different tones for 'All True' and 'RF'
palette = {
    'Review I (All True)': '#66c2a5',
    'Review I (RF)': '#1b9e77',
    'Review II (All True)': '#fc8d62',
    'Review II (RF)': '#e41a1c',
    'Review III (All True)': '#8da0cb',
    'Review III (RF)': '#377eb8'
}

# Define the hue order to ensure the correct order of colors
hue_order = [
    'Review I (All True)', 'Review I (RF)',
    'Review II (All True)', 'Review II (RF)',
    'Review III (All True)', 'Review III (RF)'
]

# Adjust the legend placement to ensure it is visible
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Plot each metric as a bar plot with two bars per review
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        data=df,
        x='Model',
        y=metric,
        hue='review_selection',
        palette=palette,
        hue_order=hue_order,
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    if metric in range_0_1:
        ax.set_ylim(0, 1)
    elif metric in range_neg1_1:
        ax.set_ylim(-1, 1)
    ax.legend().remove()  # Remove individual legends

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust legend and layout
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3)
plt.tight_layout(rect=[0, 0.95, 1, 0.95])
plt.savefig(f"{workdir}joined_analysis_RF.png", dpi=300)
plt.show()



# Adjust the legend placement to ensure it is visible
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

column_subset = {
    'Review I (All True)',
    'Review II (All True)',
    'Review III (All True)'}
df_subset_alltrue = df[df['review_selection'].isin(column_subset)]
df_subset_alltrue['review_selection'] = df_subset_alltrue['review_selection'].str.replace(' (All True)', '')
palette_alltrue = {
    'Review I': '#66c2a5',
    'Review II': '#fc8d62',
    'Review III': '#8da0cb',
}

# Plot each metric as a bar plot with two bars per review
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        data=df_subset_alltrue,
        x='Model',
        y=metric,
        hue='review_selection',
        palette=palette_alltrue,
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    if metric in range_0_1:
        ax.set_ylim(0, 1)
    elif metric in range_neg1_1:
        ax.set_ylim(-1, 1)
    ax.legend().remove()  # Remove individual legends

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust legend and layout
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3)
plt.tight_layout(rect=[0, 0.95, 1, 0.95])
plt.savefig(f"{workdir}joined_analysis.png", dpi=300)
plt.show()





# List of reviews and models
reviews = df_subset_alltrue['review'].unique()
models = ['gpt-3.5-turbo-0125', 'llama3:8b', 'mistral:v0.2']

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7))

# Plot each confusion matrix
for i, review in enumerate(reviews):
    for j, model in enumerate(models):
        ax = axes[i, j]
        sub_df = df_subset_alltrue[(df_subset_alltrue['review'] == review) & (df_subset_alltrue['Model'] == model)]
        if sub_df.empty:
            ax.set_visible(False)
            continue

        # Construct confusion matrix
        TP = sub_df['TP'].values[0]
        TN = sub_df['TN'].values[0]
        FP = sub_df['FP'].values[0]
        FN = sub_df['FN'].values[0]
        cm = np.array([[TN, FP], [FN, TP]])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,  linecolor='white', linewidths=1)
        if i == 2:
            ax.set_xlabel('Predicted True', fontsize=9)
        else:
            ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel('Actual True', fontsize=9)
        else:
            ax.set_ylabel('')
        ax.set_title(f'{model}', fontsize=9)

# Set common x-axis titles for columns
for ax, col in zip(axes[0], models):
    ax.set_title(col, fontsize=9)

# Add "Review I - III" labels to the left of the plots in reverse order
for i, review in enumerate(reversed(reviews)):
    fig.text(0.02, (2*i + 1) / 6.0, review, va='center', rotation='vertical', fontsize=9)

# Adjust layout
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

# Save the figure
plt.savefig(f"{workdir}confusion_matrices_grid.png", dpi=300)

# Show the figure
plt.show()




column_subset = {
    'Review I (RF)',
    'Review II (RF)',
    'Review III (RF)'}
df_subset_RF = df[df['review_selection'].isin(column_subset)]


# List of reviews and models
reviews = df_subset_RF['review'].unique()
models = ['gpt-3.5-turbo-0125', 'llama3:8b', 'mistral:v0.2']

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7))

# Plot each confusion matrix
for i, review in enumerate(reviews):
    for j, model in enumerate(models):
        ax = axes[i, j]
        sub_df = df_subset_RF[(df_subset_RF['review'] == review) & (df_subset_RF['Model'] == model)]
        if sub_df.empty:
            ax.set_visible(False)
            continue

        # Construct confusion matrix
        TP = sub_df['TP'].values[0]
        TN = sub_df['TN'].values[0]
        FP = sub_df['FP'].values[0]
        FN = sub_df['FN'].values[0]
        cm = np.array([[TN, FP], [FN, TP]])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,  linecolor='white', linewidths=1)
        if i == 2:
            ax.set_xlabel('Predicted True', fontsize=9)
        else:
            ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel('Actual True', fontsize=9)
        else:
            ax.set_ylabel('')
        ax.set_title(f'{model}', fontsize=9)

# Set common x-axis titles for columns
for ax, col in zip(axes[0], models):
    ax.set_title(col, fontsize=9)

# Add "Review I - III" labels to the left of the plots in reverse order
for i, review in enumerate(reversed(reviews)):
    fig.text(0.02, (2*i + 1) / 6.0, review, va='center', rotation='vertical', fontsize=9)

# Adjust layout
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

# Save the figure
plt.savefig(f"{workdir}confusion_matrices_grid_RF.png", dpi=300)

# Show the figure
plt.show()





# Set the model column as the index
df.set_index('Model', inplace=True)

# Calculate -log10(pvalue)
df['P-value'] = pd.to_numeric(df['P-value'])
df['-log10(P-value)'] = -np.log10(df['P-value'])


# Create the plot
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)  # Increased figure size

# Bar plot for Odds Ratio
bar_plot = sns.barplot(data=df, x='Model', y='Odds Ratio', hue='review_selection', palette=palette, ax=ax1)
ax1.set_ylabel('Odds Ratio', fontsize=9)
ax1.tick_params(axis='x', rotation=0)

# Remove the bar plot legend
ax1.legend_.remove()

# Create a secondary y-axis for -log10(P-value)
ax2 = ax1.twinx()

# Get the x positions of the bars
x_positions = [bar.get_x() + bar.get_width() / 2. for bar in bar_plot.patches if bar.get_height() != 0]

# Ensure x_positions matches the length of the data points
x_positions = x_positions[:len(df)]

# Plot the scatter plot with exact positions to align with the bars
ax2.scatter(x_positions, df['-log10(P-value)'], color='red', marker='o', label='-log10(p-value) Chi-Square test')

# Draw a horizontal line at -log10(0.05)
significance_level = 0.05
ax2.axhline(y=-np.log10(significance_level), color='blue', linestyle='--', linewidth=1, label='Significance level (0.05)')

# Combine legends from both plots
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

# Add a single legend at the bottom
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3)  # Adjusted bbox_to_anchor

# Add a grid for better readability
ax1.grid(True)

# Adjust layout and leave more room at the bottom
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Adjusted to leave more room at the bottom
plt.savefig(f"{workdir}odds_ratio_RF.png", dpi=300, bbox_inches='tight')

# Show the figure
plt.show()




# Set the model column as the index
df_subset_alltrue.set_index('Model', inplace=True)

# Calculate -log10(pvalue)
df_subset_alltrue['P-value'] = pd.to_numeric(df_subset_alltrue['P-value'])
df_subset_alltrue['-log10(P-value)'] = -np.log10(df_subset_alltrue['P-value'])


# Create the plot
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)  # Increased figure size

# Bar plot for Odds Ratio
bar_plot = sns.barplot(data=df_subset_alltrue, x='Model', y='Odds Ratio', hue='review_selection', palette=palette_alltrue, ax=ax1)
ax1.set_ylabel('Odds Ratio', fontsize=9)
ax1.tick_params(axis='x', rotation=0)

# Remove the bar plot legend
ax1.legend_.remove()

# Create a secondary y-axis for -log10(P-value)
ax2 = ax1.twinx()

# Get the x positions of the bars
x_positions = [bar.get_x() + bar.get_width() / 2. for bar in bar_plot.patches if bar.get_height() != 0]

# Ensure x_positions matches the length of the data points
x_positions = x_positions[:len(df_subset_alltrue)]

# Plot the scatter plot with exact positions to align with the bars
ax2.scatter(x_positions, df_subset_alltrue['-log10(P-value)'], color='red', marker='o', label='-log10(p-value) Chi-Square test')

# Draw a horizontal line at -log10(0.05)
significance_level = 0.05
ax2.axhline(y=-np.log10(significance_level), color='blue', linestyle='--', linewidth=1, label='Significance level (0.05)')

# Combine legends from both plots
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2

# Add a single legend at the bottom
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=3)  # Adjusted bbox_to_anchor

# Add a grid for better readability
ax1.grid(True)

# Adjust layout and leave more room at the bottom
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Adjusted to leave more room at the bottom
plt.savefig(f"{workdir}odds_ratio.png", dpi=300, bbox_inches='tight')
# Show the figure
plt.show()

df_subset_alltrue['Model'] = df_subset_alltrue.index
df_subset_alltrue.reset_index(drop=True, inplace=True)
test = df_subset_alltrue.groupby('Model').agg({'Odds Ratio': 'mean', 'P-value': 'mean', 'Precision': 'mean', 'Recall': 'mean', 'F1-score': 'mean', 'Matthews correlation coefficient': 'mean', 'Cohen\'s kappa': 'mean', 'PABAK': 'mean', '-log10(P-value)': 'mean'})
test[['Matthews correlation coefficient', 'PABAK']]



df_subset_RF['Model'] = df_subset_RF.index
df_subset_RF.reset_index(drop=True, inplace=True)
test2 = df_subset_RF.groupby('Model').agg({'Odds Ratio': 'mean', 'P-value': 'mean', 'Precision': 'mean', 'Recall': 'mean', 'F1-score': 'mean', 'Matthews correlation coefficient': 'mean', 'Cohen\'s kappa': 'mean', 'PABAK': 'mean'})
test2[['Matthews correlation coefficient', 'PABAK']]

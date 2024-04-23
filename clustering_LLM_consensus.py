import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_list = []
for modeltype in ['ollama', 'xinference', 'openai']:
    print("analyzing with ", modeltype)
    outdir = f'/Users/fernando/Documents/Research/aisaac/data/PICOS_physiotherapy_{modeltype}'
    df = pd.read_pickle(f'{outdir}/results.pkl')
    df['model'] = modeltype
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
df = df[['screening1', 'Population', 'Intervention', 'Control Group', 'Outcome', 'study type', 'model', 'record']]
# Define a function to calculate the majority vote

for column in ['Population', 'Intervention', 'Control Group', 'Outcome', 'study type']:
    df[column] = df[column].astype('bool')


df['screening1'] = df['screening1'].astype('category')
df['model'] = df['model'].astype('category')
df['record'] = df['record'].astype('category')
df = df.sort_values(by=['screening1', 'record', 'model'])

def majority_vote(row):
    columns = ['Population', 'Intervention', 'Control Group', 'Outcome', 'study type']
    votes = row[columns].value_counts()
    return votes.idxmax() == True

# Apply the function to each row to create the 'predicted' column
df['predicted'] = df.apply(majority_vote, axis=1)
df['predicted'] = df['predicted'].astype('bool')


# Create color map for 'record'
unique_records = df['record'].cat.codes.unique()
color_palette = sns.color_palette('hsv', len(unique_records))
record_color_map = {record: color for record, color in zip(unique_records, color_palette)}
record_colors = df['record'].cat.codes.map(record_color_map)

# Create color maps for 'screening1' and 'model'
screening1_colors = df['screening1'].cat.codes.map({0: 'red', 1: 'green'})
model_colors = df['model'].cat.codes.map({0: 'blue', 1: 'yellow', 2: 'purple'})
# Create a color map for 'predicted'
predicted_colors = df['predicted'].map({True: 'orange', False: 'cyan'})

# Create a clustermap with only column clustering
g = sns.clustermap(df.drop(['screening1', 'model', 'record', 'predicted'], axis=1),
                   row_colors=[screening1_colors, model_colors, record_colors, predicted_colors],
                   cmap='vlag',
                   figsize=(10, 10),  # Adjust the size as needed
                   col_cluster=True,
                   row_cluster=False)

# Create legends for 'screening1', 'model', and 'predicted'
for label in df['screening1'].cat.categories:
    g.ax_row_dendrogram.bar(0, 0, color='red' if label else 'green',
                            label=label, linewidth=0)
for label in df['model'].cat.categories:
    g.ax_row_dendrogram.bar(0, 0, color={'ollama': 'blue', 'xinference': 'yellow', 'openai': 'purple'}[label],
                            label=label, linewidth=0)
g.ax_row_dendrogram.bar(0, 0, color='orange', label='Predicted: Yes', linewidth=0)
g.ax_row_dendrogram.bar(0, 0, color='cyan', label='Predicted: No', linewidth=0)

# Add the legends to the plot
g.ax_row_dendrogram.legend(loc="center", ncol=2)
plt.show()
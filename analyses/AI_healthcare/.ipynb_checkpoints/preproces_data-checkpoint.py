import pandas as pd
import os


# Function to ensure .pdf suffix
def add_pdf_suffix(filename):
    if not str(filename).endswith('.pdf'):
        return str(filename) + '.pdf'
    return filename


# Usage example
file_path = '/Users/fernando/Documents/Research/academate/test_data/AI_healthcare/final_papers_bjorn.xlsx'
df = pd.read_excel(file_path)
print(df)

workdir = "/Users/fernando/Documents/Research/academate"
test_df = pd.read_pickle(f'{workdir}/test_data/AI_healthcare/preprocessed_articles_filtered.pkl')

directory =f'{workdir}/test_data/AI_healthcare/pdfs'

df['filename1'] = df.filepath.str.split('\\').str[-1]
df['filename2'] = df['file_name (without suffix)']
df['pdf_name'] = df['filename2'].replace('', None).fillna(df['filename1'])
df['pdf_name'] = df['pdf_name'].apply(add_pdf_suffix)
# Check if each file in df['pdf_name'] exists in the specified directory
df['file_exists'] = df['pdf_name'].apply(lambda x: os.path.isfile(os.path.join(directory, x)))

print(df)

pubmed2pdf_name = dict(zip(df['PUBMEDID'].astype(str), df['pdf_name']))
test_df['pdf_name'] = test_df['PUBMEDID'].astype(str).map(pubmed2pdf_name)

test_df.to_pickle(f'{workdir}/test_data/AI_healthcare/preprocessed_articles_filtered.pkl')

# Large Language Model Automation of Title and Abstract Screening in Systematic Reviews (SRs) in medicine

## Overview
This project focuses on the automation of title and abstract screening in systematic reviews (SRs) using Large Language Models (LLMs). We present a comprehensive approach encompassing data preparation, criteria implementation, quality assessment, and final classification analysis for the screening process.

## Data Preparation
- **Dataset**: Started with bibliography entries from a database search, formatted to EndNote standards.
- **Attributes**: Includes title, abstract, authors, bibliographic details, and a boolean class label for screening.
- **Text Processing**: Divided text into 5,000-character chunks with 400-character overlaps, using OpenAI's text-embedding-ada-002 model for embedding, and stored vectors in a FAISS database.

## Implementation of Selection/Exclusion Criteria
- **LLM Usage**: Employed gpt-3.5-turbo-0613 to apply original inclusion and exclusion criteria to each publication.
- **Criteria Refinement**: Modified original criteria for better LLM processing.
- **Langchain Library**: Used for prompt construction and structured responses, generating a dataset of boolean values indicating the LLM's decisions.

## Quality Assessment
- **Discriminative Ability**: Evaluated LLM's performance in separating included and excluded articles based on individual criteria.
- **Cluster Analysis**: Visual exploration of systematic review criteria application.
- **Predictive Performance**: Used PICOS criteria for binary predictions, assessing precision, recall, F1 score, and confusion matrices.
- **Random Forest Classifier**: Identified significant criteria for systematic reviews, using Gini impurity scores.

## Final Classification Analysis
- **Decision Making**: Final inclusion or exclusion of publications based on refined criteria.
- **Comparison with Human Judgments**: Examined proportion of records predicted for inclusion or exclusion against actual decisions in full text reading.
- **Odds Ratio Analysis**: Statistical measure to compare likelihoods of articles being selected for the final manuscript.

## Results
- **Insights**: Provided a detailed analysis of LLM's effectiveness in screening for SRs.
- **Statistical Significance**: Assessed through odds ratio and p-value calculations.

## Conclusion
This project demonstrates the potential of LLMs in automating the initial stages of SRs, offering a structured and statistically validated approach. The integration of advanced models and analytical techniques presents a promising direction for future research in this area.
itory.

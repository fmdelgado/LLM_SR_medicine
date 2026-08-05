# LLM screening of medical literature — code for Delgado-Chaves et al., *PNAS* (2025)

[![DOI](https://img.shields.io/badge/DOI-10.1073%2Fpnas.2411962122-blue)](https://doi.org/10.1073/pnas.2411962122)

This is the screening pipeline and the analysis code behind a peer-reviewed study that measured how well large language models perform title-and-abstract screening for systematic reviews in medicine, benchmarked against the decisions of the human reviewers who screened the same records.

> **Delgado-Chaves, F. M.**, Jennings, M. J., Atalaia, A., Wolff, J., Horvath, R., Mamdouh, Z. M., Baumbach, J., & Baumbach, L. (2025). Transforming literature screening: The emerging role of large language models in systematic reviews. *Proceedings of the National Academy of Sciences* **122**(2), e2411962122. [doi:10.1073/pnas.2411962122](https://doi.org/10.1073/pnas.2411962122)

Headline results are in the paper — please read them there rather than inferring them from this repository. BibTeX is at the bottom.

## What the pipeline does

You give it (a) a dataframe of bibliographic records — title, abstract, metadata, and the human include/exclude label from the original review — and (b) the review's inclusion/exclusion criteria written in plain English. It embeds the records into a vector store, then asks a chat model to answer **each criterion separately for each record** as a structured JSON boolean, checkpoints the answers, and merges them into a per-record decision that can be scored against the human ground truth.

**Entry point: [`new_screener1.py`](new_screener1.py)** — it defines the `Screening1` class used by every analysis here. Runs are concurrent (`asyncio`) and resumable: partial answers are checkpointed to pickles and unanswered records are retried. Filename kept as-is for continuity with the published analyses.

To reproduce a screening, run the per-review driver scripts in `new_analyses/<REVIEW>/` (three real reviews: `AI_healthcare`, `PICOS`, `PNP`); they instantiate `Screening1` once per model. Full file-by-file guide below.

## Stack, and its vintage

This is a research artifact, not a maintained product, and it reflects the state of the art at the time the study was run. The published pipeline used **`gpt-3.5-turbo`** with LangChain for prompting and structured output, OpenAI `text-embedding-ada-002` embeddings, and a **FAISS** vector store. The code as it now stands has moved to a Chroma vector store (`langchain_chroma`) and `text-embedding-3-large`, and the `new_analyses/` results directories additionally cover `gpt-4o-mini`, `gpt-4o`, and a broad set of open-weight models served via Ollama (Llama 3.x, Mistral/Mixtral, Gemma 2, Qwen 2.5, Phi-3, Granite 3, and others). The method — one criterion, one structured boolean, one record at a time, scored against human decisions — is the part that carries over; the specific models do not.

## Repository contents

| Path | What it is |
| --- | --- |
| **`new_screener1.py`** | **Main module and entry point.** Defines the `Screening1` class, which is the screening engine used by every analysis in this repository: it takes a LangChain chat model, an embedding model, a dictionary of inclusion/exclusion criteria and a dataframe of records; embeds the records into a vector store; asks the LLM to answer each criterion per record via a structured (JSON) LangChain chain; and writes the per-criterion answers, missing records and merged results to disk. Runs are resumable (partial answers are checkpointed to pickles and re-processed), and records are screened concurrently via `asyncio`. The filename is kept as-is for continuity with the published analyses. |
| `new_analyses/<REVIEW>/new_analysis_<REVIEW>_fixedparams.py` | Per-review driver scripts (`AI_healthcare`, `PICOS`, `PNP`) that define that review's criteria, instantiate `Screening1` once per model (OpenAI and Ollama-hosted open-weight models) and write results to `results_*` folders. These are the scripts to run to reproduce a screening. They resolve all paths relative to their own location, so they run from a fresh clone; set `LLM_SR_ENV_FILE` to point at an alternative `.env`. |
| `new_analyses/<REVIEW>/*_preprocessing.ipynb` | Notebooks that turn the raw bibliographic exports in `reviews_data/` into the `preprocessed_articles_filtered.pkl` inputs. |
| `new_analyses/analyze_chroma_chatcosy_fixedparams.ipynb` | Cross-review analysis notebook: metrics, agreement, random-forest criterion importance and the figures. Also exported as `.html`/`.pdf`. |
| `new_analyses/results/` | Aggregated results and plots across reviews and models. |
| `reviews_data/` | Raw bibliographic input for the three systematic reviews used here (`AI_healthcare`, `PICOS`, `PNP`). |
| `requirements.txt`, `environment.yml` | Pinned Python dependencies (pip and conda respectively). |

Running the driver scripts requires a `.env` file (git-ignored) providing the API credentials read via `os.getenv` — `openai_api` for OpenAI, and `ollama_user` / `ollama_pw` for the authenticated Ollama endpoint.

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

> **Note on `new_analyses/analyze_chroma_chatcosy_fixedparams.ipynb`:** this notebook is kept as the original analysis record and still contains the absolute paths of the machine it was executed on. The per-review driver scripts, not this notebook, are the reproducible entry points.

## Citation

If you use this code or build on this work, please cite the paper:

```bibtex
@article{delgadochaves2025transforming,
  title   = {Transforming literature screening: The emerging role of large language models in systematic reviews},
  author  = {Delgado-Chaves, Fernando M. and Jennings, Matthew J. and Atalaia, Antonio and Wolff, Justus and Horvath, Rita and Mamdouh, Zeinab M. and Baumbach, Jan and Baumbach, Linda},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {122},
  number  = {2},
  pages   = {e2411962122},
  year    = {2025},
  doi     = {10.1073/pnas.2411962122},
  url     = {https://doi.org/10.1073/pnas.2411962122}
}
```

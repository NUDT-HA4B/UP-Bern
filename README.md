# UP-Bern

## Description

This repository implements the source code for the paper **"UP-Bern: A Unified Progressive Transition Framework for Biomedical Entity Recognition and Normalization"**.

## Requirements

### Packages and Versions:
- `torch` == 1.8.1
- `transformers` == 4.5.1
- `requests` == 2.25.1
- `torchvision` == 0.9.1
- `scikit-learn` == 0.24.1
- `sqlite3`

You can install the required dependencies via:

```bash
pip install -r requirements.txt
```

## Datasets

The four datasets used in this project are:

1. **BC2GM** - [Download link](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2008-9-s2-s3
        
        
        
        )
2. **BC4CHEMD** - [Download link](https://www.sciencedirect.com/science/article/pii/S1532046413001974?via%3Dihub)
3. **BC5CDR** - [Download link](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414
        
        
        
        )
4. **NCBI** - [Download link](https://www.sciencedirect.com/science/article/pii/S1532046413001974?via%3Dihub)

These datasets are all derived from **PubMed abstracts** and are used for biomedical entity recognition tasks.

## Entity Types in Datasets

The following table outlines the different entities included in each of the four datasets:

| Dataset     | Entity Types                         |
|-------------|--------------------------------------|
| **BC2GM**   | Genes (Protein and Non-protein genes) |
| **BC4CHEMD**| Chemicals, Diseases                  |
| **BC5CDR**  | Chemicals, Diseases                  |
| **NCBI**    | Diseases, Disorders                  |

### 1. **BC2GM (BioCreative II Gene Mention)**:
   - **Entities**: Gene mentions
   - **Description**: This dataset is used for identifying **gene mentions** in biomedical text (specifically, PubMed abstracts). It contains annotated instances where gene names are mentioned in the context of scientific papers. The dataset includes both **protein** and **non-protein** gene names.

### 2. **BC4CHEMD (BioCreative IV Chemical Disease)**:
   - **Entities**: Chemicals, Diseases
   - **Description**: This dataset focuses on recognizing **chemical** and **disease** entities. The goal is to identify mentions of chemicals and diseases from PubMed abstracts and map them to their corresponding entities. It includes entities like drug names, chemical compounds, diseases, and related biomedical terminologies.

### 3. **BC5CDR (BioCreative V Chemical Disease Relationship)**:
   - **Entities**: Chemicals, Diseases
   - **Description**: This dataset is used for **chemical-disease relationship extraction**. It involves two main entity types: **chemicals** (e.g., drugs, chemical compounds) and **diseases** (e.g., medical conditions). The task is to recognize the mentions of chemicals and diseases and also predict the relationships between them.

### 4. **NCBI (National Center for Biotechnology Information Disease Corpus)**:
   - **Entities**: Diseases, Disorders
   - **Description**: The NCBI dataset is focused on **disease** mentions. It provides annotated data for recognizing **diseases** and **disorders** from scientific literature, particularly from PubMed abstracts. This dataset is useful for tasks like disease named entity recognition (NER) and classification.


## UMLS Dictionary

### Two Methods for UMLS Access:
1. **Local Database**: We provide a local SQLite database (`umls.db`). You can download it from the repository or create it yourself from the official UMLS distribution.

   - **Local Database Setup**: Follow the instructions in the `utils.py` file to set up the local database.

2. **UMLS API**: You can use the UMLS API by registering for an API key at [UMLS API](https://uts.nlm.nih.gov/). Once registered, you will need to replace the API key, username, and passphrase in `utils.py`.

### Querying Methods:
- The code allows you to choose between **local database** and **API** querying by setting `QUERY_METHOD` in the `utils.py` file.

## Search Strategies

We provide two search strategies for entity prediction:
1. **Greedy Search**: The model selects the most likely prediction at each step, generating a sequence.
2. **Beam Search**: This method uses a beam of size `k` to maintain multiple hypotheses for the best sequence of predictions.

Both strategies are implemented in `search/greedy_search.py` and `search/beam_search.py`.

## PubMedBERT

The datasets used in this work are derived from the **PubMed** corpus. The **PubMedBERT** model, a domain-specific version of BERT trained on biomedical data, is used for embedding and tokenizing PubMed abstracts.

- **PubMedBERT**: [Official Link](https://huggingface.co/allenai/pubmedbert-base)

## Citation

Coming soon...

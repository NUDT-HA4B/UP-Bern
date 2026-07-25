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

1. **BC2GM (BioCreative II)** - [Download link](https://genomebiology.biomedcentral.com/articles/10.1186/gb-2008-9-s2-s3
        
        
        
        )
2. **BC4CHEMD (BioCreative IV)** - [Download link](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S1
        
        
        
        )
3. **BC5CDR (BioCreative V)** - [Download link](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414
        
        
        
        )
4. **NCBI (National Center for Biotechnology Information Disease Corpus)** - [Download link](https://www.sciencedirect.com/science/article/pii/S1532046413001974?via%3Dihub)

These datasets are all derived from **PubMed abstracts** and are used for biomedical entity recognition tasks.

## Entity Types in Datasets

The following table outlines the different entities included in each of the four datasets:

| Dataset     | Entity Types                         |
|-------------|--------------------------------------|
| **BC2GM**   | Genes (Protein and Non-protein genes) |
| **BC4CHEMD**| Chemicals, Diseases                  |
| **BC5CDR**  | Chemicals, Diseases                  |
| **NCBI**    | Diseases, Disorders                  |

## UMLS Dictionary

The code allows you to choose between **local database** and **API** querying by setting `QUERY_METHOD` in the `utils.py` file.

1. **Local Database**: We provide a local SQLite database (`umls.db`). You can download it from the repository or create it yourself from the official UMLS distribution.

   - **Local Database Setup**: Follow the instructions in the `utils.py` file to set up the local database.

2. **UMLS API**: You can use the UMLS API by registering for an API key at [UMLS API](https://uts.nlm.nih.gov/). Once registered, you will need to replace the API key, username, and passphrase in `utils.py`.

## Search Strategies

We provide two search strategies for entity prediction:
1. **Greedy Search**: The model selects the most likely prediction at each step, generating a sequence.
2. **Beam Search**: This method uses a beam of size `k` to maintain multiple hypotheses for the best sequence of predictions.

Both strategies are implemented in `search/greedy_search.py` and `search/beam_search.py`.

## PubMedBERT

The datasets used in this work are derived from the **PubMed** corpus. The **PubMedBERT** model, a domain-specific version of BERT trained on biomedical data, is used for embedding and tokenizing PubMed abstracts.

- **PubMedBERT**: [Official Link](https://huggingface.co/allenai/pubmedbert-base)

## Citation

@inproceedings{qiu2025up,
  title={UP-Bern: A Unified Progressive Transition Framework for Biomedical Entity Recognition and Normalization},
  author={Qiu, Yanlong and Yang, Canqun and Wang, Siqi},
  booktitle={2025 IEEE International Conference on Data Mining (ICDM)},
  pages={1505--1514},
  year={2025},
  organization={IEEE}
}

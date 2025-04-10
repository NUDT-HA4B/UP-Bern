# Overview

This work utilizes several biomedical datasets and a comprehensive vocabulary to facilitate research in biomedical text mining and natural language processing (NLP). Below is an overview of the datasets and vocabulary employed:

## Datasets

### BC2GM

- **Description**: The BC2GM dataset focuses on gene mention recognition. It comprises 15,000 training sentences and 5,000 testing sentences, all sourced from PubMed abstracts. :contentReference[oaicite:0]{index=0}

### BC4CHEMD

- **Description**: The BC4CHEMD dataset contains 10,000 PubMed abstracts annotated with chemical mentions, including chemical compounds and drug names. It is divided into 6,000 abstracts for training and 2,000 each for validation and testing. :contentReference[oaicite:1]{index=1}

### BC5CDR

- **Description**: The BC5CDR dataset consists of 1,500 PubMed articles with 4,409 annotated chemicals, 5,818 diseases, and 3,116 chemical-disease interactions. :contentReference[oaicite:2]{index=2}

### NCBI Disease

- **Description**: The NCBI Disease dataset includes PubMed abstracts annotated with disease mentions, serving as a benchmark for disease entity recognition tasks. :contentReference[oaicite:3]{index=3}

**Note**: All these datasets are derived from abstracts in the PubMed database.

## Vocabulary

### UMLS 2024 Version

- **Description**: The Unified Medical Language System (UMLS) 2024 version serves as the vocabulary for this project. It is a comprehensive resource that integrates various biomedical terminologies and standards. :contentReference[oaicite:4]{index=4}

- **Size**: Approximately 5 GB.

- **Usage**: UMLS provides a standardized vocabulary that enhances the consistency and accuracy of biomedical text processing tasks.

- **download**: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html

## Usage

These datasets and the UMLS vocabulary are utilized for tasks such as named entity recognition (NER), entity normalization, and relationship extraction within biomedical texts.

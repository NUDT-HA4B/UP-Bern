
import os
import nltk
import xml.etree.ElementTree as ET
import re
from collections import defaultdict

nltk.download('punkt')

# Helper function to split text into sentences
def split_into_sentences(text):
    return nltk.sent_tokenize(text)

# BC2GM Preprocessing
def preprocess_bc2gm(data_dir):
    train_texts = []
    train_labels = []

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        sentences = f.readlines()

    with open(os.path.join(data_dir, 'train_labels.txt'), 'r') as f:
        labels = f.readlines()

    for sentence, label in zip(sentences, labels):
        sentence = sentence.strip()
        label = label.strip()

        # Split labels by spaces and extract start, end, and entity
        entities = []
        for entity in label.split('\n'):
            parts = entity.split('|')
            if len(parts) >= 3:
                entity_id, span, entity_name = parts[0], parts[1], parts[2]
                start, end = map(int, span.split())
                entities.append((start, end, entity_name))
        
        train_texts.append(sentence)
        train_labels.append(entities)

    return train_texts, train_labels


# BC4CHEMD Preprocessing
def preprocess_bc4chemd(data_dir):
    train_texts = []
    train_labels = []

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split("\t")
        abstract = parts[1]
        labels = parts[2].strip().split('\n')

        # Split the abstract into sentences
        sentences = split_into_sentences(abstract)
        for sentence in sentences:
            train_texts.append(sentence)
            
            # Parse the labels
            entities = []
            for label in labels:
                parts = label.split("\t")
                if len(parts) >= 4:
                    start, end = int(parts[2]), int(parts[3])
                    entity = parts[4]
                    entity_type = parts[5]
                    entities.append((start, end, entity, entity_type))
            train_labels.append(entities)

    return train_texts, train_labels


# BC5CDR Preprocessing
def preprocess_bc5cdr(data_dir):
    train_texts = []
    train_labels = []

    for filename in os.listdir(os.path.join(data_dir, 'train')):
        if filename.endswith('.xml'):
            tree = ET.parse(os.path.join(data_dir, 'train', filename))
            root = tree.getroot()

            # Iterate through the passages in the XML
            for passage in root.findall('passage'):
                text = passage.find('text').text
                annotations = passage.findall('annotation')

                # Split text into sentences
                sentences = split_into_sentences(text)
                for sentence in sentences:
                    train_texts.append(sentence)

                    # Collect annotations for this sentence
                    entities = []
                    for annotation in annotations:
                        entity_type = annotation.find('infon').text
                        entity_text = annotation.find('text').text
                        location = annotation.find('location')
                        start = int(location.get('offset'))
                        end = start + len(entity_text)
                        entities.append((start, end, entity_text, entity_type))
                    
                    train_labels.append(entities)

    return train_texts, train_labels


# NCBI Preprocessing
def preprocess_ncbi(data_dir):
    train_texts = []
    train_labels = []

    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split("\t")
        abstract = parts[1]
        labels = re.findall(r'<category="([^"]+)">([^<]+)</category>', abstract)

        # Split the abstract into sentences
        sentences = split_into_sentences(abstract)
        for sentence in sentences:
            train_texts.append(sentence)

            # Parse the labels
            entities = []
            for label in labels:
                category, entity = label
                start = sentence.find(entity)
                end = start + len(entity)
                entities.append((start, end, entity, category))
            train_labels.append(entities)

    return train_texts, train_labels


# Main function to preprocess the datasets
def preprocess_datasets(data_dir, dataset_name):
    if dataset_name == 'BC2GM':
        return preprocess_bc2gm(data_dir)
    elif dataset_name == 'BC4CHEMD':
        return preprocess_bc4chemd(data_dir)
    elif dataset_name == 'BC5CDR':
        return preprocess_bc5cdr(data_dir)
    elif dataset_name == 'NCBI':
        return preprocess_ncbi(data_dir)
    else:
        raise ValueError("Dataset not recognized")

# Example usage
data_dir = "/path/to/dataset"  # Path to your dataset directory
dataset_name = "BC2GM"  # Change this to the appropriate dataset

train_texts, train_labels = preprocess_datasets(data_dir, dataset_name)

# Save the results to files
with open('train_texts.txt', 'w') as f:
    for text in train_texts:
        f.write(f"{text}\n")

with open('train_labels.txt', 'w') as f:
    for labels in train_labels:
        for label in labels:
            f.write(f"{label[0]} {label[1]} {label[2]} {label[3]}\n")  # Format (start, end, entity, type)

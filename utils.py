import sqlite3
import requests
from transformers import PubMedBERTTokenizer

class Config:
    # Configuration class containing all hyperparameters and settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    MAX_SEQ_LEN = 128
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.1
    MAX_STEPS = 100
    TOP_K = 5
    VOCAB_SIZE = 30522  # for BERT-based models
    CHAR_EMBED_DIM = 50
    WORD_EMBED_DIM = 300
    EMBEDDING_DIM = 512
    HIDDEN_DIM = 256
    NUM_ACTIONS = 20  # Example number of possible actions

# Function to save the model's weights
def save_model(model, optimizer, epoch, path):
    """
    Save the model's state dictionary, optimizer, and epoch number.
    This allows you to resume training from the saved checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Function to load a model's state dictionary from a checkpoint
def load_model(model, optimizer, path):
    """
    Load the model's state dictionary, optimizer, and epoch number
    from a checkpoint to resume training.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Function to query UMLS using either a local database or an API
def query_umls(term, use_api=True, local_db_path="umls.db"):
    """
    Query UMLS using the term provided. The term is matched to its CUI
    (Concept Unique Identifier) and normalized term.
    - If use_api=True, use the UMLS API.
    - If use_api=False, query a local SQLite database.
    """
    if use_api:
        # Using UMLS API (requires valid credentials)
        api_url = f"https://uts-ws.nlm.nih.gov/restful/lookup?term={term}&apikey=YOUR_API_KEY"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()['cui'], response.json()['normalized_term']
        else:
            return None, None
    else:
        # Using a local SQLite database (example)
        conn = sqlite3.connect(local_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT cui, normalized_term FROM umls WHERE term=?", (term,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result
        else:
            return None, None

# Function to split sentences from the given text
def split_into_sentences(text):
    """
    Split a given text into sentences based on punctuation and white space.
    """
    sentences = text.split('.')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Tokenizer setup for PubMedBERT
tokenizer = PubMedBERTTokenizer.from_pretrained("microsoft/BioBERT-base")

# Function to preprocess dataset (tokenization and labeling)
def preprocess_datasets(data_dir, dataset_name):
    """
    Preprocess datasets by reading text files and generating labels.
    """
    # Reading dataset files and returning the text and labels
    with open(f"{data_dir}/{dataset_name}_texts.txt", "r") as text_file:
        texts = text_file.readlines()
    
    with open(f"{data_dir}/{dataset_name}_labels.txt", "r") as label_file:
        labels = label_file.readlines()
    
    return texts, labels
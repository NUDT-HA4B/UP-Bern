import torch
import torch.nn as nn
from transformers import PubMedBERTModel, PubMedBERTTokenizer
from torch.utils.data import Dataset, DataLoader
from search.greedy_search import GreedySearch  # Import GreedySearch class
from search.beam_search import BeamSearch  # Import BeamSearch class
from utils import query_umls

class EntityRecognitionNormalizationModel(nn.Module):
    def __init__(self, config):
        super(EntityRecognitionNormalizationModel, self).__init__()
        # Initialize PubMedBERT model for contextual embeddings
        self.bert = PubMedBERTModel.from_pretrained("microsoft/BioBERT-base")
        self.tokenizer = PubMedBERTTokenizer.from_pretrained("microsoft/BioBERT-base")

        # Embedding layers for character-level and word-level embeddings
        self.char_embedding = nn.Embedding(config.VOCAB_SIZE, config.CHAR_EMBED_DIM)
        self.word_embedding = nn.Embedding(config.VOCAB_SIZE, config.WORD_EMBED_DIM)

        # Transformer encoder for buffer
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=config.EMBEDDING_DIM, nhead=8)

        # LSTM layers for stack and action history
        self.stack_lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_DIM, num_layers=2, batch_first=True)
        self.action_lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_DIM, num_layers=2, batch_first=True)

        # Fully connected layer for action prediction
        self.fc = nn.Linear(config.HIDDEN_DIM, config.NUM_ACTIONS)
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        # Action embedding layer
        self.action_embedding = nn.Embedding(config.NUM_ACTIONS, config.ACTION_EMBEDDING_DIM)

    def max_pooling(self, x):
        return torch.max(x, dim=1)[0]

    def co_attention(self, mention, concept):
        mention_left = mention[:, :mention.size(1)//2]
        mention_right = mention[:, mention.size(1)//2:]
        concept_left = concept[:, :concept.size(1)//2]
        concept_right = concept[:, concept.size(1)//2:]

        left_attention = torch.matmul(mention_left, concept_left.transpose(1, 2))
        right_attention = torch.matmul(mention_right, concept_right.transpose(1, 2))

        attention = torch.cat([left_attention, right_attention], dim=-1)
        attention_weights = torch.softmax(attention, dim=-1)

        mention_weighted = torch.matmul(attention_weights, concept)
        return mention_weighted

    def forward(self, x):
        char_emb = self.char_embedding(x)
        word_emb = self.word_embedding(x)

        # Ensure proper padding and tokenization
        input_ids = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_SEQ_LEN)
        outputs = self.bert(**input_ids)
        last_hidden_state = outputs.last_hidden_state

        # Concatenate all embeddings (character, word, and BERT)
        combined_emb = torch.cat((char_emb, word_emb, last_hidden_state), dim=-1)

        buffer_output = self.max_pooling(combined_emb)

        stack_output, _ = self.stack_lstm(combined_emb)
        action_output, _ = self.action_lstm(combined_emb)

        # Concatenate all outputs (stack, action, buffer)
        combined_output = torch.cat((stack_output, action_output, buffer_output), dim=-1)
        combined_output = self.dropout(combined_output)

        # Predict the next action using a fully connected layer
        action_probs = self.fc(combined_output)
        return action_probs

    def greedy_search(self, x, max_steps=config.MAX_STEPS):
        # Use GreedySearch from the search folder
        search = GreedySearch(self, x)
        predictions = search.greedy_search(x, max_steps)
        
        # Regularize using UMLS
        regularized_predictions = self.regularize_with_umls(predictions)
        return regularized_predictions

    def beam_search(self, x, max_steps=config.MAX_STEPS, beam_size=5, top_k=config.TOP_K):
        # Use BeamSearch from the search folder
        search = BeamSearch(self, x, beam_size)
        predictions = search.beam_search(x, max_steps, beam_size, top_k)
        
        # Regularize using UMLS
        regularized_predictions = self.regularize_with_umls(predictions)
        return regularized_predictions

    def regularize_with_umls(self, predictions):
        # Regularize the model predictions using UMLS
        regularized = []
        for term in predictions:
            # Use the query_umls function (API or local based on config)
            cui, normalized_term = query_umls(term)
            
            if cui:
                regularized.append((term, cui, normalized_term))  # (original, CUI, normalized term)
            else:
                regularized.append((term, None, None))  # No match found

        return regularized
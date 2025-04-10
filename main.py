import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import PubMedBERTTokenizer
from module import EntityRecognitionNormalizationModel, create_data_loader, train, evaluate
from utils import Config
from utils import save_model, load_model

# Initialize configuration
config = Config()

# Dataset selection and path setup
dataset_name = "BC2GM"  # Change dataset as needed (e.g., BC2GM, BC4CHEMD, BC5CDR, NCBI)
data_dir = "/path/to/dataset"  # Dataset path

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and preprocess data
train_texts, train_labels = preprocess_datasets(data_dir, dataset_name)
train_texts_split = [split_into_sentences(abstract) for abstract in train_texts]
train_data_loader = create_data_loader(train_texts_split, train_labels, tokenizer, config.MAX_SEQ_LEN, config.BATCH_SIZE)

# Device configuration
device = config.DEVICE

# Model initialization
model = EntityRecognitionNormalizationModel(config).to(device)

# Optimizer and learning rate scheduler setup
optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
total_steps = len(train_data_loader) * config.EPOCHS
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / total_steps, 1))

# Define the model save path with dataset name
model_save_path = os.path.join(output_dir, f"{dataset_name}_model.pth")

# Training and evaluation loop
for epoch in range(config.EPOCHS):
    print(f'\nEpoch {epoch + 1} - Dataset: {dataset_name}')
    
    # Training
    print(f"Training on {dataset_name} dataset...")
    train(model, train_data_loader, optimizer, compute_loss, config, device)
    
    # Update learning rate
    scheduler.step()  
    
    # Evaluate and print precision, recall, F1 score
    print(f"Evaluating on {dataset_name} dataset...")
    precision, recall, f1 = evaluate(model, train_data_loader, config, device)
    print(f"Epoch {epoch + 1} - Precision (micro): {precision:.4f}, Recall (micro): {recall:.4f}, F1 (micro): {f1:.4f}")
    
    # Save model after each epoch
    save_model(model, optimizer, epoch, model_save_path)

# Load model (example)
model, optimizer, epoch = load_model(model, optimizer, model_save_path)
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Initialize the tokenizer and model
# Using a small, pre-trained model like distilbert for simplicity
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 2: Input text for tokenization
input_text = "Tokenization and embeddings are essential for NLP tasks."

# Step 3: Tokenize the text
encoded_input = tokenizer(
    input_text,
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,         # Pad the inputs if needed
    truncation=True,      # Truncate if the input exceeds the model's max length
    max_length=512        # Max token length for the model
)

# Step 4: Generate embeddings
with torch.no_grad():
    output = model(**encoded_input)

# The embeddings are usually the output from the last hidden layer
embeddings = output.last_hidden_state

# Step 5: Display results
print("Input Text:", input_text)
print("Original Tokens:", input_text.split())
print("Tokenized Input IDs:", encoded_input['input_ids'])
print("Tokenized Tokens:", tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0]))
print("Embeddings Shape:", embeddings.shape)

# Display a single embedding vector (e.g., the embedding for the first token)
first_token_embedding = embeddings[0, 0]  # Batch index 0, token index 0
print("First Token's Embedding Vector:", first_token_embedding.numpy())
print("First Token's Embedding Vector Size:", first_token_embedding.size())
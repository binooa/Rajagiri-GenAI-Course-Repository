# Tokenization and Embedding with Transformers

This Python script demonstrates the usage of the Hugging Face Transformers library to perform tokenization and generate embeddings for a given text input using a pre-trained language model.

## Features

1. **Pre-trained Model**:
   - Utilizes the `distilbert-base-uncased` model, a smaller and faster version of BERT, for simplicity and efficiency.

2. **Tokenization**:
   - Converts input text into token IDs using a pre-trained tokenizer.
   - Supports padding and truncation to handle varying input lengths.

3. **Embedding Generation**:
   - Extracts embeddings (numerical representations) from the last hidden layer of the pre-trained model.

4. **Output Display**:
   - Displays tokenized input IDs, the corresponding tokens, and the shape of the embeddings.
   - Outputs the embedding vector for the first token in the input sequence.

## Steps in the Script

### Step 1: Initialize the Tokenizer and Model
- The script loads the tokenizer and model for `distilbert-base-uncased` from the Hugging Face Transformers library.
- These pre-trained components are ready for immediate use without additional training.

### Step 2: Input Text
- The input text is defined as:
  ```
  "Tokenization and embeddings are essential for NLP tasks."
  ```

### Step 3: Tokenization
- The input text is tokenized using the tokenizer.
- Tokenization outputs:
  - Token IDs (numerical representation of tokens).
  - Tokens (actual words or subwords in the input text).
- Options like `padding`, `truncation`, and `max_length` ensure the tokenized input meets the model's requirements.

### Step 4: Embedding Generation
- The tokenized input is passed through the pre-trained model.
- Embeddings are extracted from the last hidden layer using:
  ```python
  output.last_hidden_state
  ```

### Step 5: Display Results
- The script prints:
  - The input text.
  - Tokenized input IDs.
  - Tokenized tokens.
  - The shape of the embeddings (dimensions).
  - The embedding vector for the first token.

## Example Output

For the input:
```text
"Tokenization and embeddings are essential for NLP tasks."
```
The script outputs:

- **Tokens**:
  ```python
 ['Tokenization', 'and', 'embeddings', 'are', 'essential', 'for', 'NLP', 'tasks.']
  ```
- **Tokenized Input IDs**:
  ```python
  [101, 22559, 1998, 15306, 2024, 6827, 2005, 17953, 13465, 1012, 102]
  ```
- **Tokenized Tokens**:
  ```python
 ['[CLS]', 'token', '##ization', 'and', 'em', '##bed', '##ding', '##s', 'are', 'essential', 'for', 'nl', '##p', 'tasks', '.', '[SEP]']
  ```
- **Embeddings Shape**:
  ```python
  torch.Size([1, 12, 768])
  ```
  - Batch size: 1
  - Sequence length: 12 tokens
  - Embedding dimension: 768
- **First Token's Embedding Vector**:
  ```python
  [0.21, -0.03, ..., 0.42]
  ```
  - A 768-dimensional vector representing the first token.

## Customization

1. **Model**:
   - Replace `distilbert-base-uncased` with any other pre-trained model from the Hugging Face Model Hub.

2. **Input Text**:
   - Modify `input_text` to analyze different sentences or documents.

3. **Embedding Extraction**:
   - Change the layer from which embeddings are extracted (e.g., intermediate layers) if needed.

4. **Output**:
   - Customize the displayed results to include more tokens or specific token embeddings.


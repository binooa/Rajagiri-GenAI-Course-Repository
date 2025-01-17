# Custom Tokenizer and Embedding Generator

This Python script implements a custom tokenizer and embedding system to process text data, simulate token embeddings, and integrate contextual information into the embeddings.

## Features

1. **Vocabulary Management**:
   - Predefined vocabulary with word-to-index mapping.
   - Dynamic expansion to include new words encountered in the input text.

2. **Custom Embeddings**:
   - Words are represented by randomly initialized embeddings of a fixed dimension.
   - Consistent embeddings ensured by setting a fixed random seed.

3. **Preprocessing**:
   - Converts text to lowercase.
   - Removes punctuation for clean tokenization.

4. **Tokenization**:
   - Splits text into tokens.
   - Maps tokens to their respective indices in the vocabulary.

5. **Contextual Embedding Integration**:
   - Adjusts word embeddings using a co-occurrence-based context window.
   - Incorporates neighboring words to enrich embedding representations.

6. **Output**:
   - Displays tokens, token IDs, and individual embeddings.
   - Prints the processed input text for clarity.

## Usage

### Prerequisites
- Python 3.x
- NumPy

### How to Run

1. Define a custom vocabulary in the script (example provided in the `vocabulary` dictionary).
2. Add or modify input text in the `input_text` variable.
3. Run the script:

   ```bash
   python script_name.py
   ```

### Example Output

For the input:
```text
"Tokenization and embeddings are essential for NLP tasks."
```

The script produces:
- **Processed Text**: Lowercased and punctuation-removed.
- **Tokens**: Extracted words.
- **Token IDs**: Indices mapped from the vocabulary.
- **Embeddings**: Numerical vectors representing each token.

Example:
```text
Input Text: Tokenization and embeddings are essential for NLP tasks.
Processed Text: tokenization and embeddings are essential for nlp tasks
Tokens: ['tokenization', 'and', 'embeddings', 'are', 'essential', 'for', 'nlp', 'tasks']
Token IDs: [1, 2, 3, 4, 5, 6, 7, 8]
Embeddings for each token:
  tokenization: [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
  and: [0.15599452 0.05808361 0.86617615 0.60111501 0.70807258]
  embeddings: [0.02058449 0.96990985 0.83244264 0.21233911 0.18182497]
  are: [0.18340451 0.30424224 0.52475643 0.43194502 0.29122914]
  essential: [0.61185289 0.13949386 0.29214465 0.36636184 0.45606998]
  for: [0.78517596 0.19967378 0.51423444 0.59241457 0.04645041]
  nlp: [0.60754485 0.17052412 0.06505159 0.94888554 0.96563203]
  tasks: [0.80839735 0.30461377 0.09767211 0.68423303 0.44015249]
```

## Customization

1. **Embedding Dimension**:
   Modify `embedding_dim` in the `CustomTokenizerAndEmbeddings` class.

2. **Stop Words**:
   Update the `stop_words` set to exclude or include specific words during context integration.

3. **Random Seed**:
   Adjust `seed` for reproducibility or randomized embeddings.


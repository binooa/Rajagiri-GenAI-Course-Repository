import string
from collections import defaultdict
import numpy as np

class CustomTokenizerAndEmbeddings:
    def __init__(self, vocabulary, embedding_dim=5, seed=42):
        """
        Initialize the custom tokenizer and embedding logic.
        - vocabulary: A dictionary mapping words to unique indices.
        - embedding_dim: The dimensionality of the word embeddings.
        - seed: A fixed seed value for reproducibility.
        """
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.seed = seed
        np.random.seed(self.seed)  # Set the random seed for reproducibility
        self.embeddings = {
            word: np.random.rand(self.embedding_dim)  # Randomly initialized embeddings
            for word in self.vocabulary.keys()
        }
        self.stop_words = {"and", "for", "the", "of", "is", "as", "to"}

    def preprocess_text(self, text):
        """
        Preprocess the text by converting it to lowercase and removing punctuation.
        """
        return text.lower().translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        """
        Tokenize the input text into words and map each word to its index in the vocabulary.
        """
        tokens = text.split()
        token_ids = [self.vocabulary.get(token, 0) for token in tokens]
        return tokens, token_ids

    def expand_vocabulary(self, tokens):
        """
        Dynamically add new words to the vocabulary and initialize their embeddings.
        """
        for token in tokens:
            if token not in self.vocabulary:
                new_index = len(self.vocabulary) + 1
                self.vocabulary[token] = new_index
                np.random.seed(self.seed + new_index)  # Ensure reproducibility for new tokens
                self.embeddings[token] = np.random.rand(self.embedding_dim)  # Randomly initialize new embeddings

    def integrate_co_occurrence_into_embeddings(self, tokens, context_window=2):
        """
        Integrate co-occurrence information into the embeddings.
        - Adjust embeddings based on the context window.
        """
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        for i, token1 in enumerate(filtered_tokens):
            context_embeddings = []
            for j in range(max(0, i - context_window), min(len(filtered_tokens), i + context_window + 1)):
                if i != j:
                    token2 = filtered_tokens[j]
                    context_embeddings.append(self.embeddings[token2])
            if context_embeddings:
                self.embeddings[token1] += np.mean(context_embeddings, axis=0)

    def get_embeddings(self, tokens):
        """
        Retrieve embeddings for the given tokens based on a simulated output similar to `output.last_hidden_state`.
        """
        token_embeddings = np.array([self.embeddings.get(token, np.zeros(self.embedding_dim)) for token in tokens])
        return token_embeddings

    def display_results(self, input_text):
        """
        Process the input text and display the results, including tokens, token IDs, and embeddings.
        """
        print("Processing the input text...")
        
        # Preprocess the text
        processed_text = self.preprocess_text(input_text)
        
        # Tokenize the text
        tokens, token_ids = self.tokenize(processed_text)
        
        # Expand vocabulary for new tokens
        self.expand_vocabulary(tokens)
        
        # Integrate co-occurrence information into embeddings
        self.integrate_co_occurrence_into_embeddings(tokens)
        
        # Retrieve embeddings
        token_embeddings = self.get_embeddings(tokens)
        
        # Display results
        print("\nInput Text:", input_text)
        print("Processed Text:", processed_text)
        print("Tokens:", tokens)
        print("Token IDs:", token_ids)
        print("Embeddings for each token:")
        for token, embedding in zip(tokens, token_embeddings):
            print(f"  {token}: {embedding}")

if __name__ == "__main__":
    # Define a custom vocabulary
    vocabulary = {
        "tokenization": 1,
        "and": 2,
        "embeddings": 3,
        "are": 4,
        "essential": 5,
        "for": 6,
        "nlp": 7,
        "tasks": 8
    }

    # Instantiate the custom tokenizer and embedding handler
    demo = CustomTokenizerAndEmbeddings(vocabulary)

    # Input text for testing
    input_text = "Tokenization and embeddings are essential for NLP tasks."

    # Display the results
    demo.display_results(input_text)

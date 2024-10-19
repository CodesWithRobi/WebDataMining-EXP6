import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Ensure you have the stopwords corpus downloaded
nltk.download('stopwords')

# Step 1: Preprocessing function (tokenization, lowercasing, stopword removal, stemming)
def preprocess_text(text):
    # Tokenization and lowercase
    tokens = text.lower().split()
    
    # Remove punctuation
    tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back into a string
    return ' '.join(tokens)

# Sample documents and queries
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A fox is quick and brown",
    "Dogs are not lazy, they are active at night",
]

# Step 2: Preprocess the documents
preprocessed_docs = [preprocess_text(doc) for doc in documents]

# Step 3: Use TF-IDF Vectorizer to convert documents into vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)  # Create a TF-IDF matrix

# Display the TF-IDF matrix
print("TF-IDF Matrix:")
print(pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()))

# Query
query = input("Query:")
preprocessed_query = preprocess_text(query)
query_vector = vectorizer.transform([preprocessed_query])

# Step 4: Compute Cosine Similarity
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Step 5: Sort documents by similarity
ranked_docs = np.argsort(cosine_similarities)[::-1]

print("\nRanked Documents based on Cosine Similarity:")
for idx in ranked_docs:
    print(f"Document {idx}: {documents[idx]} (Score: {cosine_similarities[idx]:.4f})")


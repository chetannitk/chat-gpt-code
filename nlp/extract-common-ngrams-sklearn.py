from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def extract_common_ngrams(docs, ngram_range=(1, 2), min_df=2, max_features=10):
    """
    Extracts common meaningful n-gram patterns from a list of documents.

    Args:
        docs (list): List of strings representing the documents.
        ngram_range (tuple): The range of n-gram sizes to consider (min_n, max_n).
        min_df (int or float): Minimum number of documents a token must appear in to be included.
        max_features (int): The maximum number of features (n-grams) to return.

    Returns:
        pandas.DataFrame: A DataFrame containing the most common n-grams and their frequencies.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(docs)
    ngrams = vectorizer.get_feature_names_out()
    frequencies = X.sum(axis=0).A1
    ngram_freq = dict(zip(ngrams, frequencies))
    df = pd.DataFrame(ngram_freq.items(), columns=['ngram', 'frequency'])
    df.sort_values(by='frequency', ascending=False, inplace=True)
    return df.reset_index(drop=True)

# Example usage:
documents = [
    "This is the first document. It contains some text.",
    "The second document is here. It has different content.",
    "And finally, the third document with more text."
]

ngram_range = (2, 3)  # Change this to extract different n-gram ranges (e.g., (1, 1) for unigrams, (2, 2) for bigrams, etc.)
min_df = 1  # Minimum number of documents a token must appear in to be included
max_features = 100  # Maximum number of features (n-grams) to return

result_df = extract_common_ngrams(documents, ngram_range=ngram_range, min_df=min_df, max_features=max_features)
print(result_df)

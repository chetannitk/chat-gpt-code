import nltk
import string
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')

def extract_ngrams(docs, n):
    """
    Extracts n-grams from a list of documents.

    Args:
        docs (list): List of strings representing the documents.
        n (int): The size of the n-grams to extract.

    Returns:
        list: A list of n-grams.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)

    def preprocess(text, lemmatize=False):
        # Remove punctuation and lowercase the text
        text = text.translate(translator).lower()
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        if lemmatize:
            # Lemmatize the tokens
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        else:
            lemmatized_tokens = tokens
        # Remove stopwords
        return [word for word in lemmatized_tokens if word not in stop_words]
 

    tokens = [preprocess(doc) for doc in docs]

    #tokens = [nltk.word_tokenize(doc.lower()) for doc in docs]
    n_grams = [ngrams(token, n) for token in tokens]
    return ["_".join(ng) for n_gram in n_grams for ng in n_gram]

def create_ngram_dataframe(docs, n):
    """
    Creates a DataFrame with n-gram frequencies sorted by ranking.

    Args:
        docs (list): List of strings representing the documents.
        n (int): The size of the n-grams to extract.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'ngram' and 'frequency'.
    """
    n_grams = extract_ngrams(docs, n)
    ngram_freq = Counter(n_grams)
    df = pd.DataFrame(ngram_freq.items(), columns=['ngram', 'frequency'])
    df.sort_values(by='frequency', ascending=False, inplace=True)
    return df.reset_index(drop=True)

# Example usage:
if __name__ == "__main__":
    documents = [
        "This is the first document. It contains some text.",
        "The second document is here. It has different content.",
        "And finally, the third document with more text."
    ]

    ngram_size = 3  # Change this to extract different n-grams (e.g., 1 for unigrams, 2 for bigrams, etc.)

    result_df = create_ngram_dataframe(documents, ngram_size)
    print(result_df)
    

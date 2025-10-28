import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtraction:
    def __init__(self, config):
        self.config = config

    def prepare_book_features(books_df):
        """
        Prepare book features for content-based filtering
        """
        books_features = books_df.copy()
        books_features['combined_text'] = (
            books_features['title'].fillna('') + ' ' +
            books_features['author'].fillna('') + ' ' +
            books_features['publisher'].fillna('')
        )
        books_features['combined_text'] = books_features['combined_text'].apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower())
        )
        books_features['year'] = pd.to_numeric(books_features['year'], errors='coerce')
        books_features['year'] = books_features['year'].fillna(books_features['year'].median())
        return books_features

    # Create TF-IDF vectors for book text features
    def create_tfidf_features(books_df, max_features=1000):
        """
        Create TF-IDF features from book text
        """
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        tfidf_matrix = tfidf.fit_transform(books_df['combined_text'])
        return tfidf_matrix, tfidf
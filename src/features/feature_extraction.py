import pandas as pd
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import logging


class FeatureExtraction:
    def __init__(self, config):
        self.data_processing_config = config['data_preprocessing_config']
        self.data_transformation_config = config['data_transformation_config']

    def prepare_book_features(self):
        """
        Prepare book features for content-based filtering
        """
        books_df = pickle.load(open(os.path.join(self.data_processing_config['serialized_objects_dir'], "final_rating.pkl"),'rb'))
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
        books_features = books_features.drop_duplicates('ISBN').reset_index(drop=True)

        os.makedirs(self.data_transformation_config['books_features'], exist_ok=True)
        pickle.dump(books_features, open(os.path.join(self.data_transformation_config['books_features'], "books_features.pkl"),'wb'))
        logging.info(f"Saved books_features serialization object to {self.data_transformation_config['books_features']}")
        return books_features

    # Create TF-IDF vectors for book text features
    def create_tfidf_features(self, books_df, max_features=1000):
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

        os.makedirs(self.data_transformation_config['tfidf_matrix'], exist_ok=True)
        pickle.dump(tfidf_matrix, open(os.path.join(self.data_transformation_config['tfidf_matrix'], "tfidf_matrix.pkl"), 'wb'))
        logging.info(f"Saved tfidf_matrix serialization object to {self.data_transformation_config['tfidf_matrix']}")

        os.makedirs(self.data_transformation_config['tfidf_vectorizer'], exist_ok=True)
        pickle.dump(tfidf, open(os.path.join(self.data_transformation_config['tfidf_vectorizer'], "tfidf_vectorizer.pkl"), 'wb'))
        logging.info(f"Saved tfidf_vectorizer serialization object to {self.data_transformation_config['tfidf_vectorizer']}")
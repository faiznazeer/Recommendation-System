from src.data.data_ingestion import DataIngestion
from src.data.data_processing import DataProcessing
from src.features.feature_extraction import FeatureExtraction
from src.models.recommender import ContentBasedRecommender
import os

class RecommenderPipeline:
    def __init__(self, config):
        self.config = config
        self.data_ingestion = DataIngestion(self.config['data_ingestion_config'])
        self.data_preprocessing = DataProcessing(self.config['data_preprocessing_config'])
        self.feature_extraction = FeatureExtraction(self.config)
        self.recommender_model = ContentBasedRecommender()

    def run_data_pipeline(self):
        zip_file_path = self.data_ingestion.download_data()
        self.data_ingestion.extract_zip_file(zip_file_path=zip_file_path)
        self.data_preprocessing.preprocess_data()

    def run_feature_pipeline(self):
        books_df = self.feature_extraction.prepare_book_features()
        self.feature_extraction.create_tfidf_features(books_df)

    def run_model_pipeline(self, liked_books_isbns, ratings) -> list:
        user_profile = self.recommender_model.create_user_profile(liked_books_isbns, ratings)
        recommendations = self.recommender_model.get_recommendations(user_profile)
        return recommendations

    def run_evaluation_pipeline(self):
        pass

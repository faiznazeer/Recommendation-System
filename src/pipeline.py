from src.data.data_ingestion import DataIngestion
from src.data.data_processing import DataProcessing
from src.features.feature_extraction import FeatureExtraction
from src.models.recommender import ContentBasedRecommender

class RecommenderPipeline:
    def __init__(self, config):
        self.config = config
        self.data_ingestion = DataIngestion(self.config)
        self.data_preprocessing = DataProcessing(self.config)
        self.feature_extraction = FeatureExtraction()
        self.recommender_model = ContentBasedRecommender()

    def run_data_pipeline(self):
        zip_file_path = self.data_ingestion.download_data()
        self.data_ingestion.extract_zip_file(zip_file_path=zip_file_path)
        self.data_preprocessing.data_preprocess()

    def run_feature_pipeline(self):
        self.feature_extraction.prepare_book_features()
        self.feature_extraction.create_tfidf_features()

    def run_model_pipeline(self):
        pass

    def run_evaluation_pipeline(self):
        pass

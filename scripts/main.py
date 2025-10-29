from src.pipeline import RecommenderPipeline
from src.utils.utils import read_yaml_file
import os
import pickle

class PipelineManager:
    def __init__(self, config_file_path):
        self.config = read_yaml_file(file_path=config_file_path)
        self.pipeline = RecommenderPipeline(self.config)

    def run_entire_workflow(self, user_profile):
        if not os.path.exists(self.config['data_preprocessing_config']['serialized_objects_dir']):
            data = self.pipeline.run_data_pipeline()
        
        if not (os.path.exists(self.config['data_transformation_config']['tfidf_matrix']) 
                    and os.path.exists(self.config['data_transformation_config']['tfidf_vectorizer'])):
            self.pipeline.run_feature_pipeline()

        recommendations = self.pipeline.run_model_pipeline(user_profile)
        return recommendations
    
    def run_start_up_workflow(self):
        final_rating_path = os.path.join(self.config['data_preprocessing_config']['serialized_objects_dir'], 'final_rating.pkl')
        books_features_path = os.path.join(self.config['data_transformation_config']['books_features'], 'books_features.pkl')
        tfidf_matrix_path = os.path.join(self.config['data_transformation_config']['tfidf_matrix'], 'tfidf_matrix.pkl')
        tfidf_vectorizer_path = os.path.join(self.config['data_transformation_config']['tfidf_vectorizer'], 'tfidf_vectorizer.pkl')

        if not os.path.exists(final_rating_path):
            print("**********running data pipeline**************")
            data = self.pipeline.run_data_pipeline()
        
        if not (os.path.exists(tfidf_matrix_path) and os.path.exists(tfidf_vectorizer_path)):
            print("**********running feature pipeline**************")
            self.pipeline.run_feature_pipeline()

        
        books_df = pickle.load(open(books_features_path, 'rb'))
        tfidf_matrix = pickle.load(open(tfidf_matrix_path, 'rb'))
        tfidf_vectorizer = pickle.load(open(tfidf_vectorizer_path, 'rb'))

        self.pipeline.recommender_model.set_books_df(books_df)
        self.pipeline.recommender_model.set_tfidf_matrix(tfidf_matrix)
        self.pipeline.recommender_model.set_tfidf_vectorizer(tfidf_vectorizer)

        return books_df
        
    def run_recommendation_workflow(self, liked_books_isbns, ratings):
        recommendations = self.pipeline.run_model_pipeline(liked_books_isbns, ratings)
        return recommendations
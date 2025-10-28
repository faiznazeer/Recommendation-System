import os
import pickle
import pandas as pd
from src.utils.logger import logging


class DataProcessing:
    def __init__(self, data_processing_config):
        self.data_processing_config = data_processing_config

    def preprocess_data(self):
        try:
            ratings = pd.read_csv(self.data_processing_config.ratings_csv_file, sep=";", error_bad_lines=False, encoding='latin-1')
            books = pd.read_csv(self.data_processing_config.books_csv_file, sep=";", error_bad_lines=False, encoding='latin-1')
            
            logging.info(f" Shape of ratings data file: {ratings.shape}")
            logging.info(f" Shape of books data file: {books.shape}")

            books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]
            books.rename(columns={"Book-Title":'title',
                                'Book-Author':'author',
                                "Year-Of-Publication":'year',
                                "Publisher":"publisher",
                                "Image-URL-L":"image_url"},inplace=True)

            ratings.rename(columns={"User-ID":'user_id',
                                'Book-Rating':'rating'},inplace=True)

            # Store users who had at least rated more than 50 books
            x = ratings['user_id'].value_counts() > 50
            y = x[x].index
            ratings = ratings[ratings['user_id'].isin(y)]

            ratings_with_books = ratings.merge(books, on='ISBN')
            number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
            number_rating.rename(columns={'rating':'num_of_rating'},inplace=True)
            final_rating = ratings_with_books.merge(number_rating, on='title')

            # Take books that got at least 50 rating of user
            # final_rating = final_rating[final_rating['num_of_rating'] >= 50]

            # Drop the duplicates
            final_rating.drop_duplicates(['user_id','title'],inplace=True)
            logging.info(f" Shape of the final clean dataset: {final_rating.shape}")
                        
            # Saving the cleaned data for transformation
            os.makedirs(self.data_processing_config.clean_data_dir, exist_ok=True)
            final_rating.to_csv(os.path.join(self.data_processing_config.clean_data_dir,'clean_data.csv'), index = False)
            logging.info(f"Saved cleaned data to {self.data_processing_config.clean_data_dir}")

            #saving final_rating objects for web app
            os.makedirs(self.data_processing_config.serialized_objects_dir, exist_ok=True)
            pickle.dump(final_rating,open(os.path.join(self.data_processing_config.serialized_objects_dir, "final_rating.pkl"),'wb'))
            logging.info(f"Saved final_rating serialization object to {self.data_processing_config.serialized_objects_dir}")

        except Exception as e:
            raise Error("Preprocessing data failed...\n", e)
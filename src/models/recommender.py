import numpy as np
import pandas as pd
from src.utils.utils import cosine_similarities_sparse


class ContentBasedRecommender:
    def __init__(self, books_df=None, tfidf_matrix=None, tfidf_vectorizer=None):
        self.books_df = books_df
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer

    def set_books_df(self, books_df):
        self.books_df = books_df

    def set_tfidf_matrix(self, tfidf_matrix):
        self.tfidf_matrix = tfidf_matrix

    def set_tfidf_vectorizer(self, tfidf_vectorizer):
        self.tfidf_vectorizer = tfidf_vectorizer
        
    def create_user_profile(self, liked_books_isbns, ratings=None):
        """
        Create a user profile based on books they liked
        
        Args:
            liked_books_isbns: List of ISBNs of books the user liked
            ratings: Optional list of ratings for the books (1-10 scale)
        
        Returns:
            user_profile: TF-IDF vector representing user preferences
        """
        if ratings is None:
            ratings = [1.0] * len(liked_books_isbns)  # Default equal weight
        
        # Normalize ratings to 0-1 scale
        ratings = np.array(ratings) / 10.0
        
        # Find book indices
        book_indices = []
        valid_ratings = []
        
        for isbn, rating in zip(liked_books_isbns, ratings):
            book_idx = self.books_df[self.books_df['ISBN'] == isbn].index
            if len(book_idx) > 0:
                book_indices.append(book_idx[0])
                valid_ratings.append(rating)
        
        if len(book_indices) == 0:
            print("Warning: No valid books found in the dataset")
            return None
            
        # Create weighted user profile
        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        
        for idx, rating in zip(book_indices, valid_ratings):
            user_profile += rating * self.tfidf_matrix[idx].toarray().flatten()
        
        # Normalize by number of books
        user_profile = user_profile / len(book_indices)
        
        return user_profile
    
    def get_recommendations(self, user_profile, n_recommendations=10, exclude_books=None):
        """
        Get book recommendations based on user profile
        
        Args:
            user_profile: TF-IDF vector representing user preferences
            n_recommendations: Number of recommendations to return
            exclude_books: List of ISBNs to exclude from recommendations
        
        Returns:
            recommendations: DataFrame with recommended books
        """
        if user_profile is None:
            return pd.DataFrame()
        
        # Calculate similarity between user profile and all books (sparse CPU path)
        similarities = cosine_similarities_sparse(self.tfidf_matrix, user_profile)
        
        # Create recommendations dataframe
        recommendations = self.books_df.copy()
        recommendations['similarity_score'] = similarities
        
        # Exclude books if specified
        if exclude_books:
            recommendations = recommendations[~recommendations['ISBN'].isin(exclude_books)]
        
        # Sort by similarity and return top recommendations
        recommendations = recommendations.sort_values('similarity_score', ascending=False).drop_duplicates(subset='ISBN', keep='first')
        
        return recommendations.head(n_recommendations)
    
    def get_similar_books(self, book_isbn, n_similar=5):
        """
        Get books similar to a given book
        
        Args:
            book_isbn: ISBN of the reference book
            n_similar: Number of similar books to return
        
        Returns:
            similar_books: DataFrame with similar books
        """
        # Find book index
        book_idx = self.books_df[self.books_df['ISBN'] == book_isbn].index
        if len(book_idx) == 0:
            print(f"Book with ISBN {book_isbn} not found")
            return pd.DataFrame()
        
        book_idx = book_idx[0]
        
        # Compute similarity to all books using the selected book's TF-IDF vector
        ref_vec = self.tfidf_matrix[book_idx].toarray().ravel()
        similarities = cosine_similarities_sparse(self.tfidf_matrix, ref_vec)
        
        # Create similar books dataframe
        similar_books = self.books_df.copy()
        similar_books['similarity_score'] = similarities
        
        # Exclude the reference book itself
        similar_books = similar_books[similar_books['ISBN'] != book_isbn]
        
        # Sort by similarity and return top similar books
        similar_books = similar_books.sort_values('similarity_score', ascending=False).drop_duplicates(subset='ISBN', keep='first')
        
        return similar_books.head(n_similar)

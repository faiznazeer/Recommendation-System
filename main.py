import streamlit as st
import yaml
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.recommender import ContentBasedRecommender
from scripts.main import PipelineManager
from src.utils.constant import CONFIG_FILE_PATH

# Page config
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
)


# @st.cache_data
# def load_books_raw(path: str = "notebooks/BX-Books.csv") -> pd.DataFrame:
#     df = pd.read_csv(path, sep=";", encoding="latin-1", on_bad_lines='skip')
#     return df


# def prepare_books_df(raw: pd.DataFrame) -> pd.DataFrame:
#     """Normalize column names and create combined text for TF-IDF."""
#     books = raw.copy()
#     # Rename to simpler names if present
#     rename_map = {}
#     if "Book-Title" in books.columns:
#         rename_map["Book-Title"] = "title"
#     if "Book-Author" in books.columns:
#         rename_map["Book-Author"] = "author"
#     if "Year-Of-Publication" in books.columns:
#         rename_map["Year-Of-Publication"] = "year"
#     if "Publisher" in books.columns:
#         rename_map["Publisher"] = "publisher"
#     if "Image-URL-M" in books.columns:
#         rename_map["Image-URL-M"] = "image_url"
#     if "Image-URL-L" in books.columns and "image_url" not in rename_map.values():
#         rename_map["Image-URL-L"] = "image_url"
#     if "ISBN" in books.columns and "ISBN" not in rename_map:
#         # keep ISBN as-is
#         pass

#     books = books.rename(columns=rename_map)

#     # Ensure required columns exist
#     for col in ["ISBN", "title", "author"]:
#         if col not in books.columns:
#             books[col] = ""

#     books["title"] = books["title"].fillna("")
#     books["author"] = books["author"].fillna("")
#     books["publisher"] = books.get("publisher", "").fillna("")

#     books["combined_text"] = (
#         books["title"].astype(str) + " " + books["author"].astype(str) + " " + books["publisher"].astype(str)
#     )
#     books["combined_text"] = books["combined_text"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", str(x).lower()))

#     # Reset index to have consistent indexing for TF-IDF matrix
#     books = books.reset_index(drop=True)
#     return books


# @st.cache_resource
# def build_recommender(books: pd.DataFrame, max_features: int = 1000) -> ContentBasedRecommender:
#     tfidf = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8)
#     tfidf_matrix = tfidf.fit_transform(books["combined_text"].astype(str))
#     recommender = ContentBasedRecommender(books_df=books, tfidf_matrix=tfidf_matrix, tfidf_vectorizer=tfidf)
#     return recommender


def add_selected_book(isbn: str, title: str):
    sel = st.session_state.get("selected_books", [])
    if isbn not in [b["isbn"] for b in sel] and len(sel) < 5:
        sel.append({"isbn": isbn, "title": title})
    st.session_state["selected_books"] = sel


def remove_selected_book(isbn: str):
    sel = st.session_state.get("selected_books", [])
    sel = [b for b in sel if b["isbn"] != isbn]
    st.session_state["selected_books"] = sel


def main():
    pm = PipelineManager(CONFIG_FILE_PATH)

    st.title("📚 Book Recommendation System")

    # raw_books = load_books_raw()
    # books = prepare_books_df(raw_books)
    # recommender = build_recommender(books)
    books = pm.run_start_up_workflow()

    if "selected_books" not in st.session_state:
        st.session_state["selected_books"] = []

    # Top-level layout: search + selection panel and recommendations
    left, right = st.columns([2, 1])

    with left:
        st.header("Search and select up to 5 books you know")
        query = st.text_input("Search by title or author")

        if query:
            matches = books[books["title"].str.contains(query, case=False, na=False) | books["author"].str.contains(query, case=False, na=False)]
            if matches.empty:
                st.info("No matches found")
            else:
                for _, row in matches.head(10).iterrows():
                    cols = st.columns([1, 4, 1])
                    with cols[0]:
                        if "image_url" in row and pd.notna(row.get("image_url")) and row.get("image_url"):
                            st.image(row.get("image_url"), width=64)
                    with cols[1]:
                        st.markdown(f"**{row['title']}**  \\\n+*{row['author']}*  \\\n+ISBN: {row['ISBN']}")
                    with cols[2]:
                        if st.button("Add", key=f"add_{row['ISBN']}"):
                            add_selected_book(row["ISBN"], row["title"])

        st.markdown("---")
        st.subheader("Selected books and ratings")

        sel = st.session_state.get("selected_books", [])
        if not sel:
            st.info("No books selected. Use Search to add up to 5 books.")
        else:
            for b in sel:
                cols = st.columns([4, 1, 1])
                with cols[0]:
                    st.write(f"**{b['title']}**  \\\n+ISBN: {b['isbn']}")
                with cols[1]:
                    # rating slider 1-10
                    key = f"rating_{b['isbn']}"
                    # Use a default value from session state if present, otherwise 7
                    default_val = st.session_state.get(key, 7)
                    # Create the slider with the default. Streamlit will populate session_state[key].
                    _ = st.slider("Rating", 1, 10, value=default_val, key=key)
                with cols[2]:
                    if st.button("Remove", key=f"remove_{b['isbn']}"):
                        remove_selected_book(b["isbn"])

        if st.button("Get Recommendations from Ratings"):
            sel = st.session_state.get("selected_books", [])
            if not sel:
                st.warning("Select at least one book to get recommendations")
            else:
                liked_isbns = [b["isbn"] for b in sel]
                ratings = [st.session_state.get(f"rating_{isbn}", 7) for isbn in liked_isbns]

                recommendations = pm.run_recommendation_workflow(liked_isbns, ratings)
                # user_profile = recommender.create_user_profile(liked_isbns, ratings)
                # if user_profile is None:
                #     st.error("Could not build user profile from the selected books. Check ISBNs exist in dataset.")
                # else:
                # recommendations = pm.get_recommendations(user_profile, n_recommendations=10, exclude_books=liked_isbns)
                if recommendations.empty:
                    st.info("No recommendations found.")
                else:
                    st.subheader("Recommended Books")
                    for _, rec in recommendations.iterrows():
                        cols = st.columns([1, 4])
                        with cols[0]:
                            if pd.notna(rec.get("image_url")) and rec.get("image_url"):
                                st.image(rec.get("image_url"), width=80)
                        with cols[1]:
                            st.markdown(f"**{rec['title']}**  \\\n+*{rec.get('author','')}*  \\\n+Similarity: {rec['similarity_score']:.3f}")

    with right:
        st.sidebar.header("📊 Statistics")
        st.sidebar.write(f"Total Books: {len(books):,}")
        st.sidebar.write(f"Total Authors: {books['author'].nunique():,}")
        if "year" in books.columns:
            try:
                years = pd.to_numeric(books['year'], errors='coerce')
                st.sidebar.write(f"Publication Years: {int(years.min())} - {int(years.max())}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
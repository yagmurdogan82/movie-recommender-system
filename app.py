import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation Engine")
st.markdown("### Hybrid Recommendation System: Content-Based & Collaborative Filtering")

# --- DATA LOADING & PREPARATION ---
@st.cache_data
def load_data():
    # Load the datasets
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    # Merge datasets on 'movieId'
    df = pd.merge(ratings, movies, on='movieId')
    
    # Calculate movie statistics (Rating Average & Count) for filtering
    # We aggregate by title to get the count of ratings and the mean rating
    movie_stats = df.groupby('title')['rating'].agg(['count', 'mean'])
    
    # Create the Genre Matrix for Content-Based Filtering
    # One-Hot Encoding the genres
    genre_matrix = movies['genres'].str.get_dummies(sep='|')
    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    
    return df, movies, movie_stats, cosine_sim

# Load data (Cached for performance)
df, movies, movie_stats, cosine_sim = load_data()

# --- RECOMMENDATION FUNCTIONS ---

def get_content_recommendations(movie_title, num_recommendations=5):
    """
    Suggests movies based on genre similarity (Content-Based).
    """
    # Find the index of the movie
    try:
        idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return []

    # Get similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores (Highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top N most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the titles
    return movies['title'].iloc[movie_indices].values

def get_collaborative_recommendations(movie_title, num_recommendations=5):
    """
    Suggests movies based on user rating patterns (Collaborative Filtering).
    """
    # Create the User-Item Matrix (Pivot Table)
    user_movie_ratings = df.pivot_table(index='userId', columns='title', values='rating')
    
    # Check if the movie is in the pivot table
    if movie_title not in user_movie_ratings:
        return []

    # Get the ratings for the target movie
    target_movie_ratings = user_movie_ratings[movie_title]
    
    # Calculate correlation with other movies
    similar_movies = user_movie_ratings.corrwith(target_movie_ratings)
    
    # Create a DataFrame for correlations
    corr_df = pd.DataFrame(similar_movies, columns=['Correlation'])
    corr_df.dropna(inplace=True)
    
    # Filter out noise: Keep movies with more than 50 ratings
    # We join with the movie_stats calculated earlier
    corr_df = corr_df.join(movie_stats['count'])
    recommends = corr_df[corr_df['count'] > 50].sort_values(by='Correlation', ascending=False)
    
    # Remove the movie itself from the list and return top N
    recommends = recommends.drop(movie_title, errors='ignore')
    return recommends.head(num_recommendations).index.values

# --- USER INTERFACE (UI) ---

# Filter for popular movies to make the selection list cleaner
popular_movies = movie_stats[movie_stats['count'] > 50].index.sort_values()

selected_movie = st.selectbox("Please select a movie:", popular_movies)

if st.button('Show Recommendations 🚀'):
    col1, col2 = st.columns(2)

    with col1:
        st.info("🎭 Content-Based (Similar Genres)")
        content_recs = get_content_recommendations(selected_movie)
        if len(content_recs) > 0:
            for i, movie in enumerate(content_recs, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("No data found for this movie.")

    with col2:
        st.success("👥 Collaborative (Users also liked)")
        collab_recs = get_collaborative_recommendations(selected_movie)
        if len(collab_recs) > 0:
            for i, movie in enumerate(collab_recs, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("Not enough user data to make a recommendation.")
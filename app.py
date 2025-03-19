import streamlit as st  # ✅ Import Streamlit first
import pickle
import gdown
import os

# ✅ Set Streamlit page configuration FIRST (before any other Streamlit function)
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ✅ Google Drive file ID
file_id = "16L5SgLBvi6xfs2ZgwrCc5-kQmIjPn9df"
output_path = "similarity.pkl"

# ✅ Use Streamlit's caching to avoid re-downloading
@st.cache_data  # Ensures the file is downloaded only once
def download_file():
    if not os.path.exists(output_path):
        print("🔽 Downloading similarity.pkl from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t", output_path, quiet=False)
        print("✅ Download complete!")

# ✅ Call the function to ensure the file is available
download_file()

# ✅ Load the similarity matrix
@st.cache_resource  # Cache the similarity matrix in memory
def load_similarity():
    with open(output_path, 'rb') as file:
        return pickle.load(file)

similarity = load_similarity()

print("🎯 similarity.pkl loaded successfully!")

# Load the movies DataFrame
movies_df = pickle.load(open('movies.pkl', 'rb'))

# Convert the 'title' column to a list
movies_list = movies_df['title'].tolist()

# Apply custom CSS for better UI
st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton>button {
            background-color: #ff5733;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stSelectable {
            background-color: #2c2f33;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("🎥 Movie Recommender System")
st.write("🔎 **Find movies similar to your favorite ones!**")

# Movie selection dropdown
selected_movie_name = st.selectbox("🎬 **Select a Movie:**", movies_list)

# Recommendation function
def recommend(movie_):
    if movie_ not in movies_df['title'].values:
        return ["Movie not found!"]

    # Get index of selected movie
    movie_index = movies_df[movies_df['title'] == movie_].index[0]

    # Compute similarity scores
    distances = similarity[movie_index]

    # Get top 5 similar movies (excluding itself)
    recommended_movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # Fetch movie titles
    recommended_movies_ = [movies_df.iloc[i[0]]["title"] for i in recommended_movies_list]

    return recommended_movies_

# Streamlit button to display recommendations
if st.button('🎬 Recommend'):
    recommended_movies = recommend(selected_movie_name)
    st.subheader("📌 **Recommended Movies**:")
    for movie in recommended_movies:
        st.markdown(f"✅ **{movie}**")

# Refresh button using st.rerun()
if st.button("🔄 Refresh"):
    st.rerun()

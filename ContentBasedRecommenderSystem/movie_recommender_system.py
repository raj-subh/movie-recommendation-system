import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import ast  # For safely evaluating strings containing lists

# File paths (Update these paths if necessary)
MOVIES_FILE_PATH = (
    "D:/MachineLearningProject/ContentBasedRecommenderSystem/tmdb_5000_movies.csv"
)
CREDITS_FILE_PATH = (
    "D:/MachineLearningProject/ContentBasedRecommenderSystem/tmdb_5000_credits.csv"
)

# Load datasets
movies_df = pd.read_csv(MOVIES_FILE_PATH)
credits_df = pd.read_csv(CREDITS_FILE_PATH)

# Merge both datasets on the 'title' column
movies_df = movies_df.merge(credits_df, on="title")

# Retain only the necessary columns
movies_df = movies_df[
    ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
]

# Drop missing values to ensure clean data
movies_df.dropna(inplace=True)


# Helper function to extract genre/keyword names from JSON-like format
def extract_names(json_str):
    """Extracts names from a JSON string representation of a list of dictionaries."""
    try:
        return [item["name"] for item in ast.literal_eval(json_str)]
    except (ValueError, SyntaxError):
        return []  # Return an empty list if parsing fails


# Apply the helper function to extract genres and keywords
movies_df["genres"] = movies_df["genres"].apply(extract_names)
movies_df["keywords"] = movies_df["keywords"].apply(extract_names)


# Helper function to extract top 3 cast members
def extract_top_cast(json_str, top_n=3):
    """Extracts up to 'top_n' actors from a JSON string representation of a list."""
    try:
        cast_list = [item["name"] for item in ast.literal_eval(json_str)]
        return cast_list[:top_n]  # Limit to top N actors
    except (ValueError, SyntaxError):
        return []


movies_df["cast"] = movies_df["cast"].apply(lambda x: extract_top_cast(x, top_n=3))


# Helper function to extract director from the crew column
def extract_director(json_str):
    """Extracts the director's name from the crew list."""
    try:
        for item in ast.literal_eval(json_str):
            if item["job"] == "Director":
                return [item["name"]]  # Return as a list for consistency
    except (ValueError, SyntaxError):
        return []
    return []


movies_df["crew"] = movies_df["crew"].apply(extract_director)

# Convert 'overview' column from string to list of words for better text processing
movies_df["overview"] = movies_df["overview"].apply(
    lambda x: x.split() if isinstance(x, str) else []
)

# Remove spaces in multi-word entities for better text representation
for column in ["genres", "keywords", "cast", "crew"]:
    movies_df[column] = movies_df[column].apply(
        lambda x: [word.replace(" ", "") for word in x]
    )

# Combine relevant text data into a single 'tags' column
movies_df["tags"] = (
    movies_df["overview"]
    + movies_df["genres"]
    + movies_df["keywords"]
    + movies_df["cast"]
    + movies_df["crew"]
)

# Create a refined dataset with necessary columns
refined_df = movies_df[["movie_id", "title", "tags"]].copy()

# Convert list of words into a single string
refined_df["tags"] = refined_df["tags"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else ""
)

# Convert tags to lowercase for uniformity
refined_df["tags"] = refined_df["tags"].str.lower()

# Display the first few rows of the processed dataset
# print(refined_df.head())

# Save the cleaned dataset for further use
refined_df.to_csv("processed_movies.csv", index=False)

# Initialize the stemmer
ps = PorterStemmer()


# Optimized helper function for stemming
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


# Apply stemming to 'tags' column
refined_df["tags"] = refined_df["tags"].apply(stem)

# Convert text into numerical features using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(refined_df["tags"]).toarray()

# Print the vectorized matrix
# print(vectors)
# cv.get_feature_names_out()

# Compute cosine similarity
similarity = cosine_similarity(vectors)


# Optimized recommendation function
def recommend(movie):
    # Check if the movie exists
    if movie not in refined_df["title"].values:
        print("Movie not found. Please check the title and try again.")
        return

    # Get the index of the movie
    movie_idx = refined_df[refined_df["title"] == movie].index[0]

    # Get similarity scores for the given movie
    distances = similarity[movie_idx]

    # Get top 5 most similar movie indices (excluding itself)
    similar_movies_idx = np.argsort(distances)[::-1][1:6]

    # Fetch movie titles
    recommendations = refined_df.iloc[similar_movies_idx]["title"].tolist()

    # Display recommendations
    print(f"Movies similar to '{movie}':")
    for title in recommendations:
        print(f"- {title}")


# Example Usage:
recommend("Batman Begins")

# Save similarity matrix
with open("similarity.pkl", "wb") as file:
    pickle.dump(similarity, file)

print("similarity.pkl saved successfully!")

# Save refined_df DataFrame to a pickle file
refined_df.to_pickle("movies.pkl")

print("movies.pkl has been created successfully!")

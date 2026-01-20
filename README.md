# 🎬 Hybrid Movie Recommendation Engine

A sophisticated movie recommendation system built with Python, utilizing both **Content-Based Filtering** and **Item-Based Collaborative Filtering** to provide personalized movie suggestions.

## 🚀 Overview
This project simulates the recommendation logic used by major streaming platforms. It solves the "Cold Start" problem by combining two powerful strategies:

1.  **Content-Based Filtering:**
    * Analyzes movie genres and metadata.
    * **Logic:** "If you like *Toy Story* (Animation/Comedy), you might like *Monsters, Inc.*"
    * **Tech:** Cosine Similarity on One-Hot Encoded genre matrices.

2.  **Collaborative Filtering (Item-Based):**
    * Analyzes user behavior and rating patterns.
    * **Logic:** "Users who liked *Toy Story* also strongly liked *The Incredibles* and *Star Wars*."
    * **Tech:** Pearson Correlation on a Sparse User-Item Matrix (Pivot Table).

## 🛠️ Tech Stack
* **Python 3.11+**
* **Streamlit:** For the interactive web interface.
* **Pandas & NumPy:** For efficient data manipulation and matrix operations.
* **Scikit-Learn:** For calculating Cosine Similarity.

## 📊 Dataset
The project uses the [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/):
* 100,000 ratings
* 9,000 movies
* 600 users

## 💻 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/movie-recommender.git](https://github.com/yourusername/movie-recommender.git)
    cd movie-recommender
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📸 Screenshots
* <img width="1512" height="982" alt="Ekran Resmi 2026-01-20 20 36 21" src="https://github.com/user-attachments/assets/a4b63ba2-8db0-47ad-bd4d-a02271dc6804" /> *

---
*Developed by Yağmur Doğan*

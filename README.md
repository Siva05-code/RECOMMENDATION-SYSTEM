# RECOMMENDATION-SYSTEM

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SIVAKARTHICK B

*INTERN ID*: : CT08FYO

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEEKS

*MENTOR*: NEELA SANTOSH

**Movie Recommendation System Using Content-Based and Collaborative Filtering**

### **Introduction**
A movie recommendation system is an essential component of modern streaming platforms, helping users discover relevant content based on their preferences. This project implements a hybrid recommendation system using two major approaches: **content-based filtering** and **collaborative filtering**. The system leverages **TF-IDF vectorization and cosine similarity** for content-based recommendations and **Singular Value Decomposition (SVD)** for collaborative filtering.

---
### **Tools and Technologies Used**
1. **Python** – The primary programming language for implementing the recommendation algorithms.
2. **Pandas** – Used for data manipulation and preprocessing.
3. **NumPy** – Used for numerical computations and handling arrays.
4. **Scikit-learn** – Provides tools for text processing (TF-IDF Vectorization) and cosine similarity calculations.
5. **Surprise Library** – Used for collaborative filtering with SVD-based matrix factorization.
6. **Dataset: TMDb 5000 Movies** – Contains movie metadata, including genres, overview, ratings, and vote counts.

---
### **Project Workflow**
#### **1. Data Preprocessing**
- The dataset **tmdb_5000_movies.csv** is loaded using Pandas.
- The dataset is filtered to retain only relevant columns: `id`, `title`, `overview`, `genres`, `vote_average`, and `vote_count`.
- The genres column, originally in JSON format, is converted into a readable string format using the `ast.literal_eval()` function.
- Missing values are removed to ensure data integrity.

#### **2. Content-Based Filtering**
- The `overview` text of each movie is transformed into a numerical representation using **TF-IDF Vectorization**. This step helps convert textual descriptions into numerical features.
- **Cosine similarity** is then computed between the movies based on their TF-IDF vectors.
- When a user enters a movie title, the system retrieves the most similar movies based on cosine similarity scores.

#### **3. Collaborative Filtering Using SVD**
- The collaborative filtering approach is implemented using the **Surprise library**.
- A synthetic user-item interaction matrix is created, where user ratings are randomly generated between **1-1000 users** and mapped to movie IDs and vote averages.
- The dataset is formatted using `Reader` and `Dataset` classes from Surprise.
- The **Singular Value Decomposition (SVD)** algorithm is used to learn latent features from user ratings.
- The model is trained on **80% of the dataset**, while the remaining **20% is used for evaluation**.

#### **4. User Input and Recommendations**
- The system prompts the user to enter a **movie title** to generate recommendations using the **content-based approach**.
- The system also allows the user to enter a **rating preference** and generates recommendations using the **collaborative filtering** method.

---
### **Applications of This Project**
This movie recommendation system has a wide range of applications across various domains:

1. **Streaming Platforms (Netflix, Amazon Prime, Disney+)**
   - Personalizes content recommendations for users based on viewing history and preferences.

2. **E-Commerce & Retail (Amazon, Flipkart, eBay)**
   - The same recommendation principles can be applied to suggest products based on past purchases or user behavior.

3. **Online Learning Platforms (Coursera, Udemy, edX)**
   - Course recommendations based on previous enrollments or interests.

4. **Music and Podcast Streaming Services (Spotify, Apple Music, SoundCloud)**
   - Music and podcast recommendations based on user listening history.

5. **Social Media (YouTube, TikTok, Instagram Reels)**
   - Suggests videos and posts based on user engagement and interactions.

6. **Healthcare (Medical Research & Drug Recommendations)**
   - Can be adapted for suggesting relevant medical studies or drug interactions based on patient history.

---
### **Conclusion**
This project successfully demonstrates the power of recommendation systems using **machine learning and natural language processing** techniques. By combining **content-based filtering** and **collaborative filtering**, the system provides both personalized and relevant movie recommendations to users. The integration of **TF-IDF, cosine similarity, and SVD** ensures an efficient and accurate recommendation process. The methodology used in this project can be extended to other domains beyond movies, making it a versatile and valuable implementation.


### **Output**



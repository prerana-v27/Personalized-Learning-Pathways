# Project Report: Personalized Learning Pathways — Hybrid Course Recommendation System

## 1. Overview

This project is about building a smart course recommendation system. The main goal is to help users discover online courses that are genuinely relevant to their interests and needs, cutting through the noise of thousands of available options.

The problem is that with so many courses on platforms like Coursera, Udemy, and EdX, it’s hard for learners to find the right one. Traditional search is often too broad. This project solves that by creating a **hybrid recommendation system** that understands both the *content* of the courses and the *behavior* of the users.

This is useful because it provides a personalized learning pathway, helping users find courses that match their goals, skill level, and interests, which makes learning more effective and engaging.

## 2. Data and Preprocessing

The project uses two main data files: a course catalog and a synthetic user interaction dataset.

**Course Catalog (`all_courses_cleaned.csv`)**

This dataset is a collection of over 9,600 courses from various online platforms. It was cleaned and prepared for the models.

*   **Cleaning**: Missing values in columns like `description`, `skills`, and `subject` were filled in. A single `text` column was created by combining the `title`, `description`, and `skills` to serve as a rich source for content analysis.
*   **Columns**: The key columns include:
    *   `title`: The name of the course.
    *   `description`: A summary of what the course is about.
    *   `skills`: A list of skills the course teaches.
    *   `subject`: The topic area (e.g., "data science," "musical instruments").
    *   `difficulty`: The level of the course (e.g., "Beginner," "Intermediate").
    *   `price`: The cost of the course.

**User Interactions (`interactions_synth.csv`)**

Since real user enrollment data was not available, a synthetic dataset was created. This was a crucial step because collaborative filtering models need user interaction data to work.

*   **How it was created**: The script simulates users "enrolling" in courses. It creates records of which user took which course, along with a timestamp.
*   **Why it was needed**: This data allows the system to learn patterns like "users who took Course A also tended to take Course B," which is the foundation of collaborative filtering.

## 3. Models Built

Several different models were built and tested to find the best approach. Each model looks at the problem from a different angle.

*   **Collaborative Filtering (ItemKNN & NMF)**
    *   **What it does**: These models work like "word of mouth." They don't know anything about the course content, only what users do.
    *   **ItemKNN**: Recommends courses that are similar to what a user has already taken. "Similarity" is defined by how many of the same users took both courses.
    *   **NMF (Non-negative Matrix Factorization)**: A more advanced technique that finds hidden "latent features" for both users and courses. It tries to find underlying themes or tastes that connect users to courses.

*   **Content-Based Filtering (TF-IDF cosine)**
    *   **What it does**: This model acts like a search engine. It reads the text of every course (`title`, `description`, `skills`) and recommends courses with similar text to what a user is looking for.
    *   **TF-IDF**: A technique to convert the course text into numbers (vectors) that represent the importance of each word.
    *   **Cosine Similarity**: Measures how similar the vectors of two courses are. A high score means the courses have similar content.

*   **Clustering (KMeans)**
    *   **What it does**: This model groups similar courses into "clusters" based on their content. For example, it might create a cluster for "Python for Data Science" courses. When a user likes a course, it recommends other courses from the same cluster.

*   **Supervised Ranker (Logistic Regression)**
    *   **What it does**: This is a "meta-model" that learns from the other models. It takes the scores from the content, collaborative, and cluster models as inputs and uses a Logistic Regression classifier to predict the final probability that a user will like a course. It learns to rank the best recommendations at the top.

*   **Hybrid Model (Weighted Ensemble)**
    *   **What it does**: This is the final model. It combines the predictions from the best-performing individual models (like NMF, Content, and the Supervised Ranker) using a weighted average. The weights are learned automatically to give more importance to the models that perform better, creating a final recommendation that is more accurate and robust than any single model alone.

## 4. Training and Evaluation

To measure how good the models were, we used a few standard industry metrics. We checked the top 10 recommendations (`@10`) for each model.

*   **Precision@10**: Out of the top 10 courses we recommended, what percentage were actually relevant to the user? A higher value is better.
*   **Recall@10**: Out of all the courses the user would have been interested in, what percentage did we successfully recommend in the top 10? A higher value is better.
*   **NDCG@10**: This is like Precision, but it also cares about the *order* of the recommendations. It gives a higher score if the most relevant courses are ranked at the very top (positions 1, 2, 3, etc.).

Here are the results:

| Model               | Precision@10 | Recall@10 | NDCG@10 |
| ------------------- | ------------ | --------- | ------- |
| Popularity          | 0.0050       | 0.0498    | 0.0275  |
| ItemKNN             | 0.0021       | 0.0206    | 0.0093  |
| NMF                 | 0.0242       | 0.2420    | 0.1330  |
| Content             | 0.0267       | 0.2669    | 0.1494  |
| Cluster             | 0.0166       | 0.1658    | 0.0839  |
| Supervised Ranker   | 0.0296       | 0.2961    | 0.1640  |
| **Hybrid**          | **0.0305**   | **0.3048**| **0.1712**|

As you can see, the **Hybrid** model performed the best across all metrics, proving that combining the strengths of different models leads to better overall recommendations.

## 5. Implementation Summary

The project is organized into a clear folder structure to keep the code clean and reproducible.

*   **Folder Structure**:
    *   `Data/processed/`: Contains the cleaned course catalog and synthetic user interactions.
    *   `training/`: Contains all the Python scripts for training the different models.
    *   `artifacts/`: The output directory where all the trained models (`.pkl` and `.npz` files) are saved.

*   **Python Files**:
    *   `training/collaborative_train.py`: Trains the ItemKNN and NMF models.
    *   `training/content_cluster_train.py`: Trains the TF-IDF and KMeans models.
    *   `training/supervised_ranker_train.py`: Trains the Logistic Regression ranker.
    *   `training/hybrid_train.py`: Trains the final weighted hybrid model.
    *   `app/streamlit_app.py`: The user interface for interacting with the models.

*   **Technology**: The project is built using **Python 3.13** and key libraries including `pandas` for data handling, `scikit-learn` and `numpy` for modeling, `joblib` for saving models, and `tqdm` for progress bars.

## 6. Problems Faced and Fixes

Every project has its challenges. Here are a couple of notable ones:

*   **`scikit-surprise` Installation Issues**: The popular `surprise` library for collaborative filtering had installation problems on the development machine. Instead of getting stuck, I implemented the collaborative filtering models (ItemKNN and NMF) from scratch using `scikit-learn` and `numpy`. This gave me more control and a deeper understanding of how they work.

*   **Streamlit UI Bugs**: The Streamlit application had several bugs related to caching and data alignment, which caused it to show incorrect or repetitive results. The priority was shifted to first ensure all the backend models were trained, evaluated, and working correctly. Once the models were solid, the Streamlit app was debugged and simplified to reliably serve the recommendations.

## 7. Future Plans

This project is a strong foundation, but there are many ways to improve it in the future.

*   **Improve Cold-Start and Content Matching**: For new users or niche topics, the recommendations can still be improved. Using more advanced techniques like **word embeddings** (Word2Vec, GloVe) instead of TF-IDF could help the model understand the meaning and context of course content better.

*   **Integrate Real Student Feedback**: The current system uses synthetic interaction data. The next big step would be to integrate real user data, such as course enrollments, ratings, or even click-through rates. This would make the collaborative filtering component much more powerful and accurate.

## 8. Key Learnings

This project was a great learning experience, both technically and in terms of project management.

*   **Technical Learnings**:
    *   **Data Handling**: I learned how to process and clean messy, real-world data from multiple sources.
    *   **Sparse Matrices**: I got hands-on experience working with sparse matrices from TF-IDF, which is essential for handling large-scale text data efficiently.
    *   **Model Evaluation**: I learned the importance of using the right metrics (Precision, Recall, NDCG) to evaluate a recommender system, as simple accuracy is not meaningful here.
    *   **Hybrid Blending**: I learned how to combine multiple models into a hybrid ensemble to achieve better performance than any single model.

*   **Overall Learnings**:
    *   **Debugging**: I learned that debugging is a huge part of machine learning. The issues with the Streamlit app taught me to be systematic and patient in finding the root cause of a problem.
    *   **Project Structuring**: Keeping the code organized into separate folders for data, training, and artifacts made the project much easier to manage and reproduce.
    *   **Reproducibility**: Saving trained models and having clear training scripts is key. It allows anyone to re-run the experiments and get the same results.

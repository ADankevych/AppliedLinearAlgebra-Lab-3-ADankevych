# SVD-based Movie Recommendation System

## Overview

This project demonstrates the use of Singular Value Decomposition (SVD) to perform dimensionality reduction and build a recommendation system using movie ratings data. The project is divided into three main tasks:
Implementation of SVD
Dimensionality Reduction and Visualization
Movie Recommendation System

### Part 1: SVD Implementation

#### Description
Implement a function to perform Singular Value Decomposition (SVD) of a given matrix using NumPy. The function should return three matrices 
U, Σ, and VT and verify the decomposition by reconstructing the original matrix.

Code
``` python
import numpy as np

def my_svd(matrix):
    eigenvalues_MxMT, eigenvectors_MxMT = np.linalg.eig(np.dot(matrix, matrix.T))
    U = eigenvectors_MxMT[:, np.argsort(eigenvalues_MxMT)[::-1]]

    eigenvalues_MTxM, eigenvectors_MTxM = np.linalg.eig(np.dot(matrix.T, matrix))
    V = eigenvectors_MTxM[:, np.argsort(eigenvalues_MTxM)[::-1]]

    singular_values = np.sqrt(np.maximum(eigenvalues_MTxM, 0))
    Σ = np.zeros(matrix.shape)
    Σ[:min(matrix.shape), :min(matrix.shape)] = np.diag(singular_values)

    for i in range(len(singular_values)):
        if singular_values[i] != 0:
            U[:, i] = np.dot(matrix, V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(matrix.shape[0])

    print("U: \n", U)
    print("Σ: \n", Σ)
    print("V: \n", V.T)
    print("Reconstructed matrix: \n", np.dot(U, np.dot(Σ, V.T)).round(1))

matrix = np.array([[-10.5, 2], [-57, 5.1]])
my_svd(matrix)
```
Verify the decomposition by reconstructing the original matrix and comparing it with the input.
### Part 2.1: Dimensionality Reduction and Data Visualization

#### Description
Reduce the dimensionality of the MovieLens dataset using SVD and visualize the data to analyze user and movie similarities.

Code
``` python
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Load data
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

# Preprocess data
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)
ratings_matrix_filled = ratings_matrix.fillna(2.5)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Apply SVD
U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

# Reconstruct the matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)

# Visualize user and movie embeddings
U_plot = U[:20]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U_plot[:, 0], U_plot[:, 1], U_plot[:, 2])
plt.title('Users')
plt.show()

V_plot = Vt.T[:20]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V_plot[:, 0], V_plot[:, 1], V_plot[:, 2])
plt.title('Movies')
plt.show()
```
Instructions:
- Load and preprocess the MovieLens dataset.
- Apply SVD to reduce dimensionality.
- Visualize user and movie embeddings using 3D scatter plots.

### Part 2.2: Movie Recommendation System

#### Description
Build a recommendation system by predicting user ratings for movies and suggesting top movies to users based on predicted ratings.
Code
```python
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Load data
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

# Preprocess data
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)
ratings_matrix_filled = ratings_matrix.fillna(2.5)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Apply SVD
U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

# Reconstruct the matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)

# Prepare for recommendations
predicted_ratings_only = preds_df.copy()
for row in ratings_matrix.index:
    for column in ratings_matrix.columns:
        if not np.isnan(ratings_matrix.loc[row, column]):
            predicted_ratings_only.loc[row, column] = np.nan

def recommend_movies(user_id, num_recommendations=10):
    user_row_number = user_id
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    recommendations = sorted_user_predictions.head(num_recommendations)
    movies_df = pd.read_csv('movies.csv')
    recommended_movies = movies_df[movies_df['movieId'].isin(recommendations.index)]
    return recommended_movies[['movieId', 'title', 'genres']]

user_id = 1
recommendations = recommend_movies(user_id)
print(f"Recommended movies for user {user_id}:\n", recommendations)
```

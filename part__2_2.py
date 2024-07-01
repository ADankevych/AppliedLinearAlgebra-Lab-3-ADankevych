import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=50, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=50, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
print(preds_df)

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

predicted_ratings_only = preds_df.copy()

for row in ratings_matrix.index:
    for col in ratings_matrix.columns:
        if not np.isnan(ratings_matrix.loc[row, col]):
            predicted_ratings_only.loc[row, col] = np.nan


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

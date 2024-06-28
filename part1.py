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
        U[:, i] = np.dot(matrix, V[:, i]) / singular_values[i]

    print("U: \n", U)
    print("Σ: \n", Σ)
    print("V: \n", V.T)
    print("Reconstructed matrix: \n", np.dot(U, np.dot(Σ, V.T)).round(1))


matrix = np.array([[-10.5, 2], [-57, 5.1]])
my_svd(matrix)

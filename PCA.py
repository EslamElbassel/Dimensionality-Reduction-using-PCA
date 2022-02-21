import pandas as pd
import numpy as np

# Loading data
data = pd.read_csv('emails.csv')  # your input variables data
feature_vectors = data.iloc[1:, 1:].values  

# Step1: compute the mean
mean = []
for i in range(len(feature_vectors[0])):
    m = 0
    for j in range(len(feature_vectors)):
        m += feature_vectors[j][i]
    mean.append(m / len(feature_vectors))
mean = np.array(mean)


# shifted mean
shifted_mean = []
for vector in feature_vectors:
    shifted_mean.append(vector - mean)

# Step2: compute covariance matrix
shifted_mean = np.array(shifted_mean)
covariance_matrix = 1 / len(feature_vectors) * np.dot(shifted_mean.transpose(), shifted_mean)

# Step3: compute eigen values and eigen vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print(f"{shifted_mean}")
print(f"eigen values{eigen_values} eigen vectors {eigen_vectors}")

# Step4: compute matrix Q
normalized_eigen_vectors = []
for vector in eigen_vectors:        # normalizing eigen vectors
    normalizer = np.sqrt(np.sum(np.square(vector)))
    normalized_eigen_vectors.append(vector * 1/normalizer)

matrix_Q = np.array(normalized_eigen_vectors)
print("matrix_Q: ", matrix_Q[int(len(matrix_Q)/2):])

# Step5: transform the original matrix
reduction_size = int(len(matrix_Q)/2)
new_features = np.dot(matrix_Q[int(reduction_size):], np.transpose(shifted_mean))
print('new_features: ', new_features)

# Inverse to restore original features
restoration_mean = []
for i in range(len(feature_vectors)):
    restoration_mean.append(mean)

Q_inverse = np.transpose(matrix_Q[int(reduction_size):])
original_features = np.dot(Q_inverse, new_features) + np.transpose(restoration_mean)

print("original_features: ", original_features)

from image_loading import translated_landmarks
import numpy as np
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


centered_landmarks = translated_landmarks - np.mean(translated_landmarks, axis=1, keepdims=True)


def a_factor(target_img: np.ndarray) -> float:
    ref_img = centered_landmarks[0]
    n = norm(ref_img)**2
    a = np.dot(ref_img.reshape(-1), target_img.reshape(-1))
    return a / n


def b_factor(target_img: np.ndarray) -> float:
    ref_img = centered_landmarks[0]
    n = norm(ref_img)**2
    b = np.sum(ref_img[:, 0] * target_img[:, 1] - ref_img[:, 1] * target_img[:, 0])
    return b / n


a_list, b_list = [], []
for i in range(1, len(centered_landmarks)):
    a_list.append(a_factor(centered_landmarks[i]))
    b_list.append(b_factor(centered_landmarks[i]))


a, b = np.mean(a_list), np.mean(b_list)
s = np.sqrt(a**2 + b**2)
theta = np.arctan(b / a)

def R(s, theta):
    rotation_mtx = np.array([
        [s * np.cos(theta), -s * np.sin(theta)],
        [s * np.sin(theta), s * np.cos(theta)]
    ])

    return rotation_mtx


imgs_to_rescale = centered_landmarks[1:]

imgs_rescaled = []

for i in range(len(imgs_to_rescale)):
    img_to_use = imgs_to_rescale[i]
    rescaled_points = []
    #rescaled_points = np.dot(img_to_use, R(s, theta).T)

    for j in range(len(img_to_use)):
        rescaled_points.append(np.matmul(R(s, theta), img_to_use[j]))
    imgs_rescaled.append(rescaled_points)


imgs_rescaled = [np.array(img).reshape(5, 2) for img in imgs_rescaled]

mean_data = np.mean(imgs_rescaled, axis=0)


def plot_rescaled() -> plt.figure:
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    for i in range(5):
        axs[0].scatter(imgs_rescaled[0][i, 0], imgs_rescaled[0][i, 1], color="red")
        axs[0].scatter(imgs_rescaled[1][i, 0], imgs_rescaled[1][i, 1], color="g")
        axs[0].scatter(imgs_rescaled[2][i, 0], imgs_rescaled[2][i, 1], color="b")
        axs[0].scatter(centered_landmarks[0][i, 0], centered_landmarks[0][i, 1], marker="x", color="violet", s=100)
        axs[0].scatter(mean_data[i, 0], mean_data[i, 1], marker="s", color="brown", s=100)
        axs[1].scatter(imgs_to_rescale[0][i, 0], imgs_to_rescale[0][i, 1], color="red")
        axs[1].scatter(imgs_to_rescale[1][i, 0], imgs_to_rescale[1][i, 1], color="g")
        axs[1].scatter(imgs_to_rescale[2][i, 0], imgs_to_rescale[2][i, 1], color="b")
        axs[1].scatter(centered_landmarks[0][i, 0], centered_landmarks[0][i, 1], marker="x", color="violet", s=100)
        axs[1].scatter(mean_data[i, 0], mean_data[i, 1], marker="s", color="brown", s=100)
    axs[0].set_title("Rescaled")
    axs[1].set_title("Not Rescaled")
    return fig


z1, z2, z3, z4 = centered_landmarks[0], imgs_rescaled[0], imgs_rescaled[1], imgs_rescaled[2]
z1, z2, z3, z4 = z1.reshape(-1, 1), z2.reshape(-1, 1), z3.reshape(-1, 1), z4.reshape(-1, 1)
z_ = np.array([z1, z2, z3, z4])
z_norm = [MinMaxScaler().fit_transform(z) for z in z_]
z_ = np.asarray(z_norm)

mean_data_reshaped = mean_data.reshape(-1, 1)
mean_data_reshaped = MinMaxScaler().fit_transform(mean_data_reshaped)


deviations = z_.reshape(10, 4) - mean_data_reshaped


s_ = z_.reshape(10, 4).shape[0]
covariance_matrix = (deviations.T @ deviations) / (s_ - 1)
print("Covariance matrix:\n", covariance_matrix)

e_values, e_vectors = eig(covariance_matrix)

print("Eigenvalues:\n", e_values)
print("Eigenvectors:\n",e_vectors)
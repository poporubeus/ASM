import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from procrusters_analysis import aligned_shapes, x_init_expanded


tot_array = np.concatenate((x_init_expanded, aligned_shapes), axis=0)
#x_mean = mean_array

# 10 is the flatten dimension of the vectors, 7 is the number of images I have
flat_total_arr = np.zeros([10, 7])
for i in range(tot_array.shape[0]):
    flat_arr = tot_array[i].flatten()
    flat_total_arr[:, i] = flat_arr


# Guess the number of modes or components to express your data.
# ofc less data leads to less k components I guess..
pca = PCA(n_components=5)
pca.fit(flat_total_arr.T)

print(pca.components_) ## eigenvectors
print(pca.explained_variance_) ## eigenvalues

exp_var = pca.explained_variance_ratio_


def plot_histogram(exp_var: np.ndarray) -> plt.Figure:
    fig = plt.figure()
    plt.bar(range(1, exp_var.shape[0] + 1), exp_var, color="lightblue", edgecolor="darkblue")
    plt.xlabel(r"$n_{component}$", fontsize=15, fontweight="bold")
    plt.ylabel(r"$\mathcal{E}_{variance}$", fontsize=15, fontweight="bold")
    plt.xticks(range(1, exp_var.shape[0] + 1), fontsize=13)
    plt.yticks(fontsize=13)
    return fig

"""cov_matrix = np.cov(flat_total_arr.T, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]


k = 2
selected_eigenvectors = eigenvectors_sorted[:, :k]

data_transformed = np.dot(flat_total_arr.T, selected_eigenvectors)
print("Eigenvalues:\n", eigenvalues_sorted)
print("\nSelected Eigenvectors:\n", selected_eigenvectors)
print("\nTransformed Data:\n", data_transformed)"""

"""plt.scatter(flat_total_arr[:, 0], flat_total_arr[:, 1], color='blue', label='Original Data')
plt.scatter(data_transformed[:, 0], data_transformed[:, 1], color='red', label='Transformed Data (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.legend()
plt.show()"""
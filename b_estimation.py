import matplotlib.pyplot as plt
from procrusters_analysis import mean_array, aligned_shapes
from pca_analysis import eig_vals, eig_vecs


xm = mean_array

P = eig_vecs
### by looking at the explained variance I can use the first eigenvector since has the 60% of explainability compared
# to the others which have 20% and 10% and so on...

important_p = P[0]
x = aligned_shapes[1] # x which I want to approximate already in the training set
b = important_p.T * (x.flatten() - xm.flatten())

approx = xm.flatten() + b * important_p
print("Approximation:\n", approx.reshape((5,2)))
print("Original:\n", x)


def plot_approximation() -> plt.Figure:
    fig = plt.figure()
    plt.scatter(approx.reshape((5,2))[:, 0], approx.reshape((5,2))[:, 1], c='orange', label="Reconstructed")
    plt.scatter(x[:, 0], x[:, 1], c='royalblue', label="Original data")
    plt.legend()
    return fig
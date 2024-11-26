import numpy as np
from PIL import Image
import os
#import cv2
import matplotlib.pyplot as plt



folder_path = "/Users/francescoaldoventurelli/Desktop/fotoooo/star_pics/"
stored_imgs = []

for filename in os.listdir(folder_path):
    file = os.path.join(folder_path, filename)
    img = np.asarray(Image.open(file))
    stored_imgs.append(img)


def get_image(index: int) -> plt.figure:
    img_selected = stored_imgs[index]
    fig = plt.figure()
    plt.imshow(img_selected)
    return fig


img0_numpy = np.array([
    [300, 2050],
    [750, 3400],
    [2120, 3250],
    [2800, 1790],
    [1580, 1230],
])

img1_numpy = np.array([
    [360, 1400],
    [650, 3180],
    [2000, 3180],
    [2600, 1700],
    [1520, 900],
])

img2_numpy = np.array([
    [1050, 1480],
    [780, 2350],
    [1350, 2820],
    [2000, 2450],
    [2000, 1520],
])

img3_numpy = np.array([
    [830, 1480],
    [1450, 2500],
    [2430, 2300],
    [2800, 1160],
    [1680, 600],
])


def plot_single_img_scatter(img_point: np.ndarray) -> plt.show():
    for i in range(len(img_point)):
        plt.scatter(img_point[i, 0], img_point[i, 1])
    return plt.show()


example_data = [img0_numpy, img1_numpy, img2_numpy, img3_numpy]

#### PLOT multiple figures in different plots #####

def multi_plot() -> plt.figure:
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig = plt.figure()
    for i in range(2):
        for j in range(2):
            axs[i, j].scatter(example_data[i * 2 + j][:, 0], example_data[i * 2 + j][:, 1])
            axs[i, j].set_title(f"Star {i * 2 + j + 1}")
            axs[i, j].imshow(stored_imgs[i * 2 + j])

    return fig


test_image = stored_imgs[-1]

def flatten_img(img: np.ndarray) -> np.ndarray:
    return img.flatten()


stacked_imgs = np.vstack([img for img in example_data])

xc, yc = np.mean(stacked_imgs[:, 0]), np.mean(stacked_imgs[:, 1])

'''z0, z1, z2, z3 = (flatten_img(img0_numpy), flatten_img(img1_numpy),
                   flatten_img(img2_numpy), flatten_img(img3_numpy))'''


# ***** PLOT w centroid *****
def plot_centroids() -> plt.figure:
    fig = plt.figure()
    plt.imshow(stored_imgs[0])
    plt.imshow(stored_imgs[1], alpha=0.5)
    plt.imshow(stored_imgs[2], alpha=0.4)
    plt.imshow(stored_imgs[3], alpha=0.3)
    plt.scatter(xc, yc, s=50, c='r', marker='x')

    return fig

# ***************************


centroid = np.array([
    [xc, yc]
])

# print(centroid.shape) it is a (1, 2) vector which I use to translate all the (1, 2) images

all_imgs_centroids = np.zeros(shape=(len(example_data), 2))
for i in range(len(example_data)):
    all_imgs_centroids[i, 0] = np.mean(example_data[i][:, 0])
    all_imgs_centroids[i, 1] = np.mean(example_data[i][:, 1])


distance_translation = [(all_imgs_centroids[i]-centroid) for i in range(len(all_imgs_centroids))]
all_imgs_centroids_translated = [all_imgs_centroids[i] - distance_translation[i] for i in range(len(all_imgs_centroids))]


all_imgs_centroids_translated = np.array(all_imgs_centroids_translated).reshape(4, 2)


def plot_centralized_figures() -> plt.figure:

    fig = plt.figure()
    for i in range(all_imgs_centroids_translated.shape[0]):
        plt.scatter(all_imgs_centroids_translated[i, 0], all_imgs_centroids_translated[i, 1], c="cyan", edgecolors='royalblue')
    plt.scatter(xc, yc, s=50, c='r', marker='x')
    plt.imshow(stored_imgs[0])
    plt.imshow(stored_imgs[1], alpha=0.5)
    plt.imshow(stored_imgs[2], alpha=0.4)
    plt.imshow(stored_imgs[3], alpha=0.3)

    return fig


### procrusters analysis
# Translation

### translate all the landmarks

translated_landmarks = []
for i in range(len(example_data)):
    d = distance_translation[i]
    landmark_tr = np.array(example_data[i] - d)
    translated_landmarks.append(landmark_tr)

print("Translated landmark set:\n", translated_landmarks[0])
#print("Original image:\n", example_data[0])

fig, axs = plt.subplots(1, 2, figsize=(14, 8))
for i in range(len(translated_landmarks)):
    axs[0].scatter(translated_landmarks[i][:, 0], translated_landmarks[i][:, 1])
    axs[0].imshow(stored_imgs[i], alpha=0.5)
axs[0].set_title("Translated")
for i in range(len(example_data)):
    axs[1].scatter(example_data[i][:, 0], example_data[i][:, 1])
    axs[1].imshow(stored_imgs[i], alpha=0.5)
axs[1].set_title("Non-Translated")

plt.tight_layout()
plt.show()
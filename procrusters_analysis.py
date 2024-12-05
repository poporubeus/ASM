import matplotlib.pyplot as plt
from utils import *


### Write down some images
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


test_img = np.array([
    [350, 2580],
    [1530, 3000],
    [2600, 2300],
    [720, 1030],
    [2180, 1060]])


img4_numpy = np.array([
    [880, 1400],
    [1550, 2400],
    [2350, 2400],
    [2500, 1260],
    [1600, 800],
])

img5_numpy = np.array([
    [1000, 1600],
    [1550, 2300],
    [2200, 2000],
    [2300, 1200],
    [1550, 880],
])


img6_numpy = np.array([
    [1200, 1400],
    [1450, 2000],
    [2100, 1800],
    [1900, 1300],
    [1370, 1090],
])


x_arr = np.array([img0_numpy, img1_numpy, img2_numpy, img3_numpy,
                  img4_numpy, img5_numpy, img6_numpy], dtype=np.float32)


### translate each sample to its center of gravity
def find_centroid(img: np.ndarray) -> np.ndarray:
    c = np.mean(img, axis=0, dtype=np.float32)
    return c


x_arr_translated = np.copy(x_arr)
for i in range(len(x_arr)):
    centroid = find_centroid(x_arr[i])
    x_arr_translated[i] -= centroid


x_init = x_arr_translated[0].flatten()
x_init /= norm(x_init)
x_init = x_init.reshape((5, 2))
x_arr_translated = x_arr_translated[1:]

aligned_shapes = []
for img_to_align in x_arr_translated:
    aligned_img = np.zeros_like(img_to_align)
    a1 = a_factor(x_init, img_to_align)
    b1 = b_factor(x_init, img_to_align)
    s, theta = np.sqrt(a1**2 + b1**2), np.arctan2(b1, a1)
    r_factor = R(s, theta)
    for k in range(img_to_align.shape[0]):
        shape_old = img_to_align[k]
        shape_new = shape_old @ r_factor
        aligned_img[k] = shape_new
    aligned_img /= norm(aligned_img)
    aligned_shapes.append(aligned_img)

aligned_shapes = np.array(aligned_shapes)
x_init_expanded = np.expand_dims(x_init, axis=0)
mean_array = np.concatenate((x_init_expanded, aligned_shapes), axis=0).mean(axis=0)


def plot() -> plt.Figure:
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].scatter(x_init[:, 0], x_init[:, 1], label="Reference Shape", color="red", marker="x", s=500)
    axs[0].scatter(mean_array[:, 0], mean_array[:, 1], label="Mean Shape", color="green", marker="*", s=500)

    for i, aligned_img in enumerate(aligned_shapes):
        axs[0].scatter(aligned_img[:, 0], aligned_img[:, 1], label=f"Aligned Shape {i+1}", alpha=0.7, color="b")
    for j, aligned_img in enumerate(x_arr_translated):
        axs[1].scatter(aligned_img[:, 0], aligned_img[:, 1], label=f"Aligned Shape {j+1}", alpha=0.7, color="b")

    axs[0].legend()
    axs[1].legend()
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    return fig

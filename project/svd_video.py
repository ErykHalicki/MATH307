import numpy as np
import cv2
from matplotlib import pyplot as plt

# i am more familiar with opencv, so i will use it instead of scipy.misc.imread()

gray_image = cv2.imread("cow.jpeg", cv2.IMREAD_GRAYSCALE) # read in grayscale

print(f"shape of the image is {gray_image.shape}. Containing {gray_image.size} pixels")

cached_svd = None

def get_image_partial_svd(image, k=None):
    global cached_svd
    if cached_svd is None or image is not gray_image:
        cached_svd = np.linalg.svd(image) # U, Sigma, V.T
    u,s,vt = cached_svd

    if k is not None:
        u = u[:,:k] # get all columns of u up to column k
        vt = vt[:k] # get all rows of v up to row k
        s = s[:k] # get all diagonals of Sigma up to index k
    return u, s, vt

def reconstruct_image_from_svd(u, s, vt):
    return u@np.diag(s)@vt

def compress_and_reconstruct_image(image, k):
    u, s, vt = get_image_partial_svd(image, k)
    image = reconstruct_image_from_svd(u, s, vt)
    return (u,s,vt), image

fig, ax = plt.subplots()
img_plot = ax.imshow(gray_image, cmap='gray')
ax.axis('off')

k_direction = 1
k=2
MULTIPLIER = 1

try:
    while True:
        components, reconstructed_image = compress_and_reconstruct_image(gray_image, k)
        img_plot.set_data(reconstructed_image)
        total_datapoints = components[0].size + components[1].size + components[2].size
        ax.set_title(f"SVD image (k={k}): compression ratio {round(float(total_datapoints) / gray_image.size * 100,2)}% smallest sv: {round(np.min(components[1]),2)}")
        plt.pause(0.001)
        if k >= min(gray_image.shape[0], gray_image.shape[1]) or k <= 1:
            k_direction*=-1 
        k+=k_direction*MULTIPLIER
except Exception:
    plt.close()

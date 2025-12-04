import numpy as np
import cv2
from matplotlib import pyplot as plt

# i am more familiar with opencv, so i will use it instead of scipy.misc.imread()

gray_image = cv2.imread("eryk.jpg", cv2.IMREAD_GRAYSCALE) # read in grayscale

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

NROWS = 4
NCOLS = 3
fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS, figsize=(12, 8))
i=1

axes[0,0].imshow(gray_image, cmap="gray")
axes[0,0].set_title(f"Original image: M x N = {gray_image.size} datapoints", fontsize=10)
axes[0,0].axis('off')

def visualize_compressed_image(image,k):
    global i
    if i//NCOLS > NROWS:
        raise Exception("Used up all slots in the figure!")

    components, reconstructed_image = compress_and_reconstruct_image(image, k)
    total_datapoints = components[0].size + components[1].size + components[2].size

    axes[i//NCOLS,i%NCOLS].imshow(reconstructed_image, cmap="gray")
    axes[i//NCOLS,i%NCOLS].set_title(f"SVD image (k={k}): compression ratio {round(gray_image.size / float(total_datapoints),2)}x" , fontsize=10)
    axes[i//NCOLS,i%NCOLS].axis('off')
    i+=1

space_remaining = True
k=1
while space_remaining:
    try:
        visualize_compressed_image(gray_image, k)
        k*=2
        k = int(k)
    except Exception: 
        space_remaining = False

plt.tight_layout()
plt.show()


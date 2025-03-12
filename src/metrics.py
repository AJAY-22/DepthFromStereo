import cv2
from skimage.metrics import structural_similarity as compare_ssim

def compute_ssim(imageA, imageB, data_range):
    """
    Compute the Structural Similarity (SSIM) index between two images.

    Parameters:
        imageA (numpy.ndarray): The first image.
        imageB (numpy.ndarray): The second image.

    Returns:
        float: The SSIM index between the two images.
    """
    # Determine if images are grayscale or color and compute accordingly.
    ssim_index, _ = compare_ssim(imageA, imageB, full=True, data_range=data_range)
    return ssim_index


if __name__ == "__main__":
    # Example usage: reading two images and computing their SSIM.
    image_path1 = "bottle_l.jpg"
    image_path2 = "bottle_l.jpg"

    imageA = cv2.imread(image_path1)
    imageB = cv2.imread(image_path2)

    if imageA is None or imageB is None:
        print("Error: Could not load one or both images.")
    else:
        similarity = compute_ssim(imageA, imageB, 225)
        print("SSIM:", similarity)
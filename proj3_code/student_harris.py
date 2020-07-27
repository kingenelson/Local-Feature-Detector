import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
import pdb


def get_gaussian_kernel(ksize, sigma):
    """
    Generate a Gaussian kernel to be used in get_interest_points for calculating
    image gradients and a second moment matrix.
    You can call this function to get the 2D gaussian filter.

    This might be useful:
    2) Make sure the value sum to 1
    3) Some useful functions: cv2.getGaussianKernel

    Args:
    -   ksize: kernel size
    -   sigma: kernel standard deviation

    Returns:
    -   kernel: numpy nd-array of size [ksize, ksize]
    """

    kernel = None
    #############################################################################
    # TODO: YOUR GAUSSIAN KERNEL CODE HERE                                      #
    #############################################################################

    kernel = cv2.getGaussianKernel(ksize, sigma)
    kernel = kernel@kernel.T
    kernel = kernel/kernel.sum()

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return kernel

def my_filter2D(image, filt, bias = 0):
    """
    Compute a 2D convolution. Pad the border of the image using 0s.
    Any type of automatic convolution is not allowed (i.e. np.convolve, cv2.filter2D, etc.)

    Helpful functions: cv2.copyMakeBorder

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   filt: filter that will be used in the convolution

    Returns:
    -   conv_image: image resulting from the convolution with the filter
    """
    conv_image = None

    #############################################################################
    # TODO: YOUR MY FILTER 2D CODE HERE                                         #
    #############################################################################

    b = int((filt.shape[0]-1)/2), int((filt.shape[1]-1)/2)
    imb = cv2.copyMakeBorder(image,b[0],b[0],b[1],b[1],cv2.BORDER_CONSTANT)
    imb_i = np.argwhere(imb == image)
    conv_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            w = imb[i:i+1+2*b[0],j:j+1+2*b[1]]
            conv_image[i,j] = np.multiply(filt, w).sum()

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return conv_image

def get_gradients(image):
    """
    Compute smoothed gradients Ix & Iy. This will be done using a sobel filter.
    Sobel filters can be used to approximate the image gradient

    Helpful functions: my_filter2D from above

    Args:
    -   image: A numpy array of shape (m,n) containing the image


    Returns:
    -   ix: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the x direction
    -   iy: numpy nd-array of shape (m,n) containing the image convolved with differentiated kernel in the y direction
    """

    ix, iy = None, None
    #############################################################################
    # TODO: YOUR IMAGE GRADIENTS CODE HERE                                      #
    #############################################################################

    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    ix, iy = my_filter2D(image, sobel[0]), my_filter2D(image, sobel[1])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return ix, iy


def remove_border_vals(image, x, y, c, window_size = 16):
    """
    Remove interest points that are too close to a border to allow SIFTfeature
    extraction. Make sure you remove all points where a window around
    that point cannot be formed.

    Args:
    -   image: image: A numpy array of shape (m,n,c),
        image may be grayscale of color (your choice)
    -   x: A numpy array of shape (N,) containing the x coordinate of each pixel
    -   y: A numpy array of shape (N,) containing the y coordinate of each pixel
    -   c: A numpy array of shape (N,) containing the confidences of each pixel
    -   window_size: int of the window size that we want to remove. (i.e. make sure all
        points in a window_size by window_size area can be formed around a point)
        (set this to 16 for unit testing). treat the center point of this window as the bottom right
        of the center most 4 pixels. This will be the same window used for SIFT.

    Returns:
    -   x: A numpy array of shape (N-border_vals_removed,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N-border_vals_removed,) containing y-coordinates of interest points
    -   c: numpy array of shape (N-border_vals_removed,) containing the confidences of each pixel
    """

    #############################################################################
    # TODO: YOUR REMOVE BORDER VALS CODE HERE                                   #
    #############################################################################

    idx_lx = np.argwhere(x>=window_size/2)
    idx_ux = np.argwhere(x<=image.shape[0]-window_size/2+1)
    idx_ly = np.argwhere(y>=window_size/2)
    idx_uy = np.argwhere(y<=image.shape[0]-window_size/2+1)
    idx = np.union1d(np.intersect1d(idx_lx, idx_ux, assume_unique=True), np.intersect1d(idx_ly, idx_uy, assume_unique=True))
    x, y, c = x[idx], y[idx], c[idx]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, c

def second_moments(ix, iy, ksize = 7, sigma = 10):
    """
    Given image gradients, ix and iy, compute sx2, sxsy, sy2 using a gaussian filter.

    Helpful functions: my_filter2D

    Args:
    -   ix: numpy nd-array of shape (m,n) containing the gradient of the image with respect to x
    -   iy: numpy nd-array of shape (m,n) containing the gradient of the image with respect to y
    -   ksize: size of gaussian filter (set this to 7 for unit testing)
    -   sigma: deviation of gaussian filter (set this to 10 for unit testing)

    Returns:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: A numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None
    #############################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                       #
    #############################################################################

    k = get_gaussian_kernel(ksize, sigma)
    sx2 = my_filter2D(ix**2, k)
    sy2 = my_filter2D(iy**2, k)
    sxsy = my_filter2D(ix*iy, k)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return sx2, sy2, sxsy

def corner_response(sx2, sy2, sxsy, alpha):

    """
    Given second moments calculate corner resposne.
    R = det(M) - alpha(trace(M)^2)
    where M = [[Sx2, SxSy],
                [SxSy, Sy2]]


    Args:
    -   sx2: A numpy nd-array of shape (m,n) containing the second moment in the x direction twice
    -   sy2: A numpy nd-array of shape (m,n) containing the second moment in the y direction twice
    -   sxsy: numpy nd-array of dim (m,n) containing the second moment in the x then the y direction
    -   alpha: empirical constant in Corner Resposne equaiton (set this to 0.05 for unit testing)

    Returns:
    -   R: Corner response score for each pixel
    """

    R = None
    #############################################################################
    # TODO: YOUR CORNER RESPONSE CODE HERE                                       #
    #############################################################################

    detM = sx2 * sy2 - sxsy**2
    trM = sx2 + sy2
    R = detM - alpha * trM**2

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R

def non_max_suppression(R, neighborhood_size = 7):
    """
    Implement non maxima suppression. Take a matrix and return a matrix of the same size
    but only the max values in a neighborhood are non zero. We also do not want local
    maxima that are very small as well so remove all values that are below the median.

    Helpful functions: scipy.ndimage.filters.maximum_filter

    Args:
    -   R: numpy nd-array of shape (m, n)
    -   ksize: int that is the size of neighborhood to find local maxima (set this to 7 for unit testing)

    Returns:
    -   R_local_pts: numpy nd-array of shape (m, n) where only local maxima are non-zero
    """

    R_local_pts = None

    #############################################################################
    # TODO: YOUR NON MAX SUPPRESSION CODE HERE                                  #
    #############################################################################

    temp1 = np.where(R >= np.median(R),R,0)
    temp2 =  maximum_filter(temp1,size=neighborhood_size)
    R_local_pts = np.where(temp1 == temp2,temp2,0)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return R_local_pts


def get_interest_points(image, n_pts = 1500):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.


    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   n_pts: integer of number of interest points to obtain

    Returns:
    -   x: A numpy array of shape (n_pts) containing x-coordinates of interest points
    -   y: A numpy array of shape (n_pts) containing y-coordinates of interest points
    -   R_local_pts: A numpy array of shape (m,n) containing cornerness response scores after
            non-maxima suppression and before removal of border scores
    -   confidences: numpy nd-array of dim (n_pts) containing the strength
            of each interest point
    """
    x, y, R_local_pts, confidences = None, None, None, None


    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
    #############################################################################

    i = get_gradients(image)
    s = second_moments(i[0], i[1])
    r = corner_response(s[0], s[1], s[2], 0.05)
    R_local_pts = non_max_suppression(r)
    confidences = R_local_pts.ravel()
    pts = confidences.nonzero()[0].size
    if n_pts < pts:
        pts = n_pts
    indx = np.flip(np.argsort(confidences)[-pts:])
    y, x = np.unravel_index(indx,R_local_pts.shape)
    confidences = confidences[indx]
    x, y, confidences = remove_border_vals(image, x, y, confidences)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return x,y, R_local_pts, confidences



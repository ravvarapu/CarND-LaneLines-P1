import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    gray_image = grayscale(image)

    kernel_size = 5
    blur_image = gaussian_blur(gray_image, kernel_size)

    low_threshold = 10
    high_threshold = 150
    edge_image = canny(blur_image, low_threshold, high_threshold)

    # ROI
    ysize = image.shape[0]
    xsize = image.shape[1]
    p2 = (xsize/2 - 10, ysize/2)#top, left 
    p1 = (xsize/2 + 10, ysize/2)#top, right
    p3 = (50, ysize) #bottom, left
    p4 = (xsize - 50, ysize)

    vertices = np.array([[p1, p2, p3, p4]], dtype=np.int32)
    roi_image = region_of_interest(edge_image, vertices)

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 60 #minimum number of pixels making up a line
    max_line_gap = 20   # maximum gap in pixels between connectable line segments
    lines_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    lines_overlayed_image = weighted_img(lines_image, image)

    debug_enabled = True
    if debug_enabled:
        _, axis = plt.subplots(2, 3)
        axis[0, 0].imshow(image)
        axis[0, 1].imshow(gray_image, 'gray')
        axis[0, 2].imshow(blur_image, 'gray')
        axis[1, 0].imshow(edge_image, 'gray')
        axis[1, 1].imshow(roi_image, 'gray')
        axis[1, 2].imshow(lines_overlayed_image)
        plt.tight_layout()
        plt.show()

    return lines_overlayed_image

def process_image_params(image, params):
    gray_image = grayscale(image)

    kernel_size = params['kernel_size']
    blur_image = gaussian_blur(gray_image, kernel_size)

    low_threshold = params['low_threshold']
    high_threshold = params['low_threshold']
    edge_image = canny(blur_image, low_threshold, high_threshold)

    # ROI
    ysize = image.shape[0]
    xsize = image.shape[1]
    p2 = (xsize/2 - 10, ysize/2)#top, left 
    p1 = (xsize/2 + 10, ysize/2)#top, right
    p3 = (50, ysize) #bottom, left
    p4 = (xsize - 50, ysize)

    vertices = np.array([[p1, p2, p3, p4]], dtype=np.int32)
    roi_image = region_of_interest(edge_image, vertices)

    rho = params['rho'] # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = params['threshold']     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = params['min_line_len'] #minimum number of pixels making up a line
    max_line_gap = params['max_line_gap']   # maximum gap in pixels between connectable line segments
    lines_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    lines_overlayed_image = weighted_img(lines_image, image)

    debug_enabled = True
    if debug_enabled:
        _, axis = plt.subplots(3, 2, figsize=(20, 20))
        axis[0, 0].imshow(image)
        axis[0, 1].imshow(gray_image, 'gray')
        axis[1, 0].imshow(blur_image, 'gray')
        axis[1, 1].imshow(edge_image, 'gray')
        axis[2, 0].imshow(roi_image, 'gray')
        axis[2, 1].imshow(lines_overlayed_image)
    return lines_overlayed_image

def interact_function(img_path, kernel_size = 5, low_threshold = 10, high_threshold = 150, \
        rho = 2, threshold = 15, min_line_len = 60, max_line_gap = 20):
    params = dict()
    params['kernel_size'] = kernel_size
    params['low_threshold'] = low_threshold
    params['low_threshold'] = high_threshold
    params['rho'] = rho
    params['threshold'] = threshold
    params['min_line_len'] = min_line_len
    params['max_line_gap'] = max_line_gap
    test_data_dir = "test_images/"
    img_path = os.path.join(test_data_dir, img_path)
    image = mpimg.imread(img_path)
    process_image_params(image, params)



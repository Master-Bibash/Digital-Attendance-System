def rgb_to_grayscale(rgb_image):
    """
    Convert an RGB image to grayscale.
    
    :param rgb_image: A 3D list of RGB pixel values [[(R, G, B), ...], ...]
    :return: A 2D list of grayscale pixel values [[gray, ...], ...]
    """
    height = len(rgb_image)
    width = len(rgb_image[0])
    
    grayscale_image = []
    
    for y in range(height):
        grayscale_row = []
        for x in range(width):
            r, g, b = rgb_image[y][x]
            # Convert RGB to grayscale using the luminosity method
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_row.append(gray)
        grayscale_image.append(grayscale_row)
    
    return grayscale_image

def apply_smoothing(grayscale_image):

    height = len(grayscale_image)
    width = len(grayscale_image[0])
    
    smoothed_image = [[0 for _ in range(width)] for _ in range(height)]
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            sum_gray = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    sum_gray += grayscale_image[y + dy][x + dx]
            smoothed_image[y][x] = sum_gray // 9
    
    return smoothed_image

def histogram_equalization(grayscale_image):
    """
    Apply histogram equalization to enhance the contrast of the grayscale image.
    
    :param grayscale_image: A 2D list of grayscale pixel values [[gray, ...], ...]
    :return: A 2D list of equalized grayscale pixel values [[gray, ...], ...]
    """
    height = len(grayscale_image)
    width = len(grayscale_image[0])
    
    # Compute the histogram
    histogram = [0] * 256
    for row in grayscale_image:
        for gray in row:
            histogram[gray] += 1
    
    # Compute the cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]
    
    # Normalize the CDF
    cdf_min = min(cdf)
    cdf_max = max(cdf)
    if cdf_max == cdf_min:
        return grayscale_image  # Avoid division by zero
    
    # Apply histogram equalization
    equalized_image = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            gray = grayscale_image[y][x]
            equalized_image[y][x] = int((cdf[gray] - cdf_min) / (cdf_max - cdf_min) * 255)
    
    return equalized_image
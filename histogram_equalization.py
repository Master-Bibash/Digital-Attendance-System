def histogram_equalization(grayscale_image):

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

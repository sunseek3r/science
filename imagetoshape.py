from PIL import Image, ImageFilter


def imageToShape(image):
    return image.filter(ImageFilter.FIND_EDGES)


def findPoints(image):
    column, row = image.size
    pixels = []
    coords = []
    for i in range(row):
        for j in range(column):
            x = image.getpixel((j, i))
            if x[3] != 0:
                pixels.append(x)
                coords.append((j, i))
    return pixels, coords


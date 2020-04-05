from PIL import Image
import numpy as np
import time
from functools import wraps


def print_timing(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        start = time.perf_counter()
        result = func(*arg, **kwargs)
        end = time.perf_counter()
        fs = '{} took {:.3f} milliseconds'
        print(fs.format(func.__name__, (end - start) * 1000))
        return result
    return wrapper


def halftone(image):
    pixels = image.load()
    new_image = Image.new('L', image.size)
    new_pixels = new_image.load()
    for h in range(image.size[0]):
        for w in range(image.size[1]):
            new_pixels[h, w] = (round((pixels[h, w][0] + pixels[h, w][1] + pixels[h, w][2]) / 3))
    return new_image


def aperture_to_array(pixels, x, y, a):
    # (x, y) represents the top left corner of the window, where (0, 0) is at the top left of the image
    # a - the side of the window
    arr = np.zeros((a, a))
    for shift_x in range(a):
        for shift_y in range(a):
            #print(x, shift_x, y, shift_y)
            arr[shift_x][shift_y] = int(pixels[x + shift_x, y + shift_y])
    return arr


@print_timing
def calculate_gradients(image):
    Gx_mask = np.array([[-1, -1, -1, -1, -1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1]])
    Gy_mask = np.rot90(Gx_mask, 1)

    size0 = image.size[0] - 4
    size1 = image.size[1] - 4
    Gx = np.empty((size0, size1), dtype=int)
    Gy = np.empty((size0, size1), dtype=int)
    G = np.empty((size0, size1), dtype=int)

    pixels = image.load()
    max_g = 0
    for h in range(size0):
        for w in range(size1):
            arr = aperture_to_array(pixels, h, w, 5)
            Gx[h][w] = np.sum(np.multiply(Gx_mask, arr))
            Gy[h][w] = np.sum(np.multiply(Gy_mask, arr))
            G[h][w] = abs(Gx[h][w]) + abs(Gy[h][w])

    return (Gx, Gy, G)


def binarize(G, size, threshold):
    image = Image.new('1', size)
    pixels = image.load()
    for h in range(size[0] - 4):
        for w in range(size[1] - 4):
            if G[h][w] > threshold:
                pixels[h, w] = 1
            else:
                pixels[h, w] = 0
    return image


def normalize(array):
    min_ = 255 * 5
    max_ = -255 * 5
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            max_ = max(max_, array[i][j])
            min_ = min(min_, array[i][j])
    max_ = abs(max_ - min_)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j] = (array[i][j] - min_) * 255 / max_
    return array


def array_to_image(array):
    print(array)
    image = Image.new('L', array.shape)
    pixels = image.load()
    for h in range(array.shape[0]):
        for w in range(array.shape[1]):
            pixels[h, w] = int(array[h][w])
    return image


def grads_to_images(grads):
    images = []
    for i in range(3):
        images.append(array_to_image(normalize(grads[i])))
    return images



@print_timing
def outline(image, image_name, threshold):
    grads = calculate_gradients(image)
    grad_images = grads_to_images(grads)
    grad_images[0].save("Data/lab3/Gx_" + image_name)
    grad_images[1].save("Data/lab3/Gy_" + image_name)
    grad_images[2].save("Data/lab3/G_" + image_name)

    return binarize(grads[2], image.size, threshold)


def process_image(image_name, threshold):
    img = Image.open("Data/lab3/" + image_name)
    halftone_img = halftone(img)
    halftone_img.save("Data/lab3/halftone_" + image_name)
    outline(halftone_img, image_name, threshold).save("Data/lab3/outlined_" + image_name)


def run_test():
    process_image("img1.bmp", 63)
    process_image("img2.bmp", 115)
    process_image("img3.bmp", 75)
    process_image("img4.bmp", 65)


run_test()

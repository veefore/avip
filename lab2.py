from PIL import Image
import numpy as np
import time
from functools import wraps

def print_timing(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        start = time.perf_counter()  # needs python3.3 or higher
        result = func(*arg, **kwargs)
        end = time.perf_counter()
        fs = '{} took {:.3f} milliseconds'
        print(fs.format(func.__name__, (end - start) * 1000))
        return result
    return wrapper


def aperture_to_array(pixels, x, y):
    # (x, y) represents the top left corner of the window, where (0, 0) is at the top left of the image
    arr = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]
    for shift_x in range(3):
        for shift_y in range(3):
            #print(x, shift_x, y, shift_y)
            arr[shift_x][shift_y] = int(pixels[x + shift_x, y + shift_y] / 255)
    return arr


def erase_isolated_pixels(image):
    # image.mode == '1' is asserted
    new_image = image.copy()
    pixels = new_image.load()
    B1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    B2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    for h in range(image.size[0] - 2):
        for w in range(image.size[1] - 2):
            arr = aperture_to_array(pixels, h, w)
            if np.array_equal(B1, arr):
                pixels[h + 1, w + 1] = (0)
            elif np.array_equal(B2, arr):
                pixels[h + 1, w + 1] = (1)
    return new_image


def erase_edge_pixels(image):
    # image.mode == '1' is asserted
    A = np.array([[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
                  [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                  [[0, 0, 0], [0, 1, 0], [1, 1, 1]]])

    B1 = A
    for arr in A:
        for rot_cnt in range(1, 3):
            np.append(B1, np.array([np.rot90(arr, rot_cnt)]), 0)

    B2 = 1 - B1


    new_image = image.copy()
    pixels = new_image.load()
    for h in range((image.size[0] - 2)):
        for w in range((image.size[1] - 2)):
            ape_arr = aperture_to_array(pixels, h, w)
            flag = False
            for arr in B1:
                if np.array_equal(ape_arr, arr):
                    pixels[h + 1, w + 1] = (0)
                    flag = True
                    break
            if not flag:
                for arr in B2:
                    if np.array_equal(ape_arr, arr):
                        pixels[h + 1, w + 1] = (1)
                        break;

    return new_image


@print_timing
def erase_fringe(image):
    new_image = erase_edge_pixels(erase_isolated_pixels(image))
    return new_image


def difference_image(image1, image2):
    # assumes image1.size == image2.size
    pixels1 = image1.load()
    pixels2 = image2.load()
    image = Image.new('1', image1.size)
    pixels = image.load()
    for h in range(image1.size[0]):
        for w in range(image1.size[1]):
            pixels[h, w] = abs(pixels1[h, w] - pixels2[h, w])
    return image


def run_test():
    # img1
    img1 = Image.open("Data/lab2/img1.bmp").resize((640, 360))
    img1.save("Data/lab2/downsampled_img1.bmp")
    filtered_img1 = erase_fringe(img1)
    filtered_img1.save("Data/lab2/filtered_img1.bmp")
    difference_image(img1, filtered_img1).save("Data/lab2/diff_img1.bmp")


    # img2
    img2 = Image.open("Data/lab2/img2.bmp").resize((400, 600))
    img2.save("Data/lab2/downsampled_img2.bmp")
    filtered_img2 = erase_fringe(img2)
    filtered_img2.save("Data/lab2/filtered_img2.bmp")
    difference_image(img2, filtered_img2).save("Data/lab2/diff_img2.bmp")


    # img3
    img3 = Image.open("Data/lab2/img3.bmp").resize((300, 300))
    img3.save("Data/lab2/downsampled_img3.bmp")
    filtered_img3 = erase_fringe(img3)
    filtered_img3.save("Data/lab2/filtered_img3.bmp")
    difference_image(img3, filtered_img3).save("Data/lab2/diff_img3.bmp")


run_test()

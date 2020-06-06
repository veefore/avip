from PIL import Image
import numpy as np
from utils import print_timing


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
def outline(image, image_name, threshold, folder_path=None):
    grads = calculate_gradients(image)
    grad_images = grads_to_images(grads)
    if folder_path is not None:
        grad_images[0].save(folder_path + "Gx_" + image_name)
        grad_images[1].save(folder_path + "Gy_" + image_name)
        grad_images[2].save(folder_path + "G_" + image_name)
    return binarize(grads[2], image.size, threshold)


def process_image(image_name, threshold, folder_path=None, save=False):
    img = Image.open(folder_path + image_name)
    halftone_img = halftone(img)
    if save:
        halftone_img.save(folder_path + "halftone_" + image_name)
    outline(halftone_img, image_name, threshold, folder_path).save(folder_path + "outlined_" + image_name)


def run_test():
    folder_path = "Data/lab3/"
    process_image("img1.bmp", 63, folder_path)
    process_image("img2.bmp", 115, folder_path)
    process_image("img3.bmp", 75, folder_path)
    process_image("img4.bmp", 65, folder_path)


#run_test()

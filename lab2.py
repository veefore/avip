from PIL import Image
import numpy as np


def aperture_to_array(pixels, x, y):
    # (x, y) represents top left corner, where (0, 0) is at top left of the image
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
    pixels = image.load()
    B1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    B2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    for h in range(image.size[0] - 2):
        for w in range(image.size[1] - 2):
            arr = aperture_to_array(pixels, h, w)
            if np.array_equal(B1, arr):
                pixels[h + 1, w + 1] = (0)
            elif np.array_equal(B2, arr):
                pixels[h + 1, w + 1] = (1)
    return image


def erase_edge_pixels(image):
    # image.mode = '1' is asserted
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


    pixels = image.load()
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

    return image


def erase_fringe(image):
    for i in range(10):
        image = erase_edge_pixels(erase_isolated_pixels(image))
    return image

def negate_image(image):
    pixels = image.load()
    for h in range(image.size[0]):
        for w in range(image.size[1]):
            pixels[h, w] = 1 - pixels[h, w]
    return image


img1 = Image.open("Images/lab2/img1.bmp").resize((640, 360))
img1.save("Images/lab2/downsampled_img1.bmp")
filtered_img1 = erase_fringe(img1)
filtered_img1.save("Images/lab2/filtered_img1.bmp")

negated_img1 = negate_image(img1)
negated_img1.save("Images/lab2/negated_img1.bmp")
filtered_negated_img1 = erase_fringe(negated_img1)
filtered_negated_img1.save("Images/lab2/filtered_negated_img1.bmp")

img2 = Image.open("Images/lab2/img2.bmp").resize((400, 600))
img2.save("Images/lab2/downsampled_img2.bmp")
filtered_img2 = img2
filtered_img2 = erase_fringe(img2)
filtered_img2.save("Images/lab2/filtered_img2.bmp")

negated_img2 = negate_image(img2)
negated_img2.save("Images/lab2/negated_img2.bmp")
filtered_negated_img2 = erase_fringe(negated_img2)
filtered_negated_img2.save("Images/lab2/filtered_negated_img2.bmp")

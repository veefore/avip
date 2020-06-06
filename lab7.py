from PIL import Image
import numpy as np
from utils import print_timing
from lab1 import greyscale


def run_down(image, w, h):
    pixels = image.load()
    if h > 0 and pixels[w, h - 1] == pixels[w, h]:
        return pixels[w, h], 0

    cnt = 0
    for shift in range(image.size[1] - h):
        if pixels[w, h + shift] == pixels[w, h]:
            cnt += 1
        else:
            break
    return pixels[w, h], cnt


def run_diag(image, w, h):
    pixels = image.load()
    if w > 0 and h > 0 and pixels[w - 1, h - 1] == pixels[w, h]:
        return pixels[w, h], 0

    cnt = 0
    for shift in range(min(image.size[0] - w, image.size[1] - h)):
        if pixels[w + shift, h + shift] == pixels[w, h]:
            cnt += 1
        else:
            break
    return pixels[w, h], cnt


def run_right(image, w, h):
    pixels = image.load()
    if w > 0 and pixels[w - 1, h] == pixels[w, h]:
        return pixels[w, h], 0

    cnt = 0
    for shift in range(image.size[0] - w):
        if pixels[w + shift, h] == pixels[w, h]:
            cnt += 1
        else:
            break
    return pixels[w, h], cnt


@print_timing
def get_rl_matrix(image, length_range=None):
    assert image.mode == 'L'
    if length_range is None:
        length_range = int((image.size[0] + image.size[1]) / 2)
    matrix = np.zeros(shape=(256, length_range), dtype=int)
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            runs = [run_down(image, w, h), run_diag(image, w, h), run_right(image, w, h)]
            for pair in runs:
                matrix[pair[0], pair[1]] += 1
    return matrix


def visualize_matrix(matrix):
    image = Image.new('L', size=matrix.shape)
    max_ = np.max(matrix)
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            pixels[w, h] = int(matrix[w, h] / max_ * 255)
    return image


def count_runs(rl_matrix):
    return np.sum(rl_matrix)


def gray_level_nonuniformity(rl_matrix, k=None):
    if k is None:
        k = count_runs(rl_matrix)
    val = 0
    for length in range(rl_matrix.shape[1]):
        temp = 0
        for brightness in range(rl_matrix.shape[0]):
            temp += rl_matrix[brightness, length]
        val += temp ** 2
    return val / k


@print_timing
def process_image(filepath, length_range):
    image = Image.open(filepath)
    rl_matrix = get_rl_matrix(image, length_range)
    image = visualize_matrix(rl_matrix)
    dot = filepath.rfind('.')
    image.save(filepath[:dot] + "_matrix.bmp")
    runs = count_runs(rl_matrix)
    glnu = gray_level_nonuniformity(rl_matrix, runs)
    print("Glnu", filepath, glnu)


def run():
    folder_path = "Data/lab7/"
    for i in range(1, 7, 1):
        process_image(folder_path + "img" + str(i) + ".bmp", 80)


run()
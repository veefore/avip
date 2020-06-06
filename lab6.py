from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import print_timing
from lab4 import generate_alphabet_features, calculate_features
from lab5 import segment_line
from math import sqrt
from time import sleep


@print_timing
def images_from_coords(image, coords):
    result = []
    for rect in coords:
        result.append(image.crop(rect))
    return result


def get_distance(lhs, rhs):
    d = 'density'
    rcom = 'relative_centre_of_mass'
    rhim = 'relative_horizontal_inertia_moment'
    rvim = 'relative_vertical_inertia_moment'
    rdim = 'relative_diagonal_inertia_moment'
    rrdim = 'relative_reversed_diagonal_inertia_moment'
    rfmim = 'relative_first_main_inertia_moment'
    rsmim = 'relative_second_main_inertia_moment'

    dist = ((lhs[d] - rhs[d]) / max(lhs[d], rhs[d])) ** 2
    dist += ((lhs[rcom][0] - rhs[rcom][0]) / max(lhs[rcom][0], rhs[rcom][0])) ** 2
    dist += ((lhs[rcom][1] - rhs[rcom][1]) / max(lhs[rcom][1], rhs[rcom][1])) ** 2
    dist += ((lhs[rhim] - rhs[rhim]) / max(lhs[rhim], rhs[rhim])) ** 2
    dist += ((lhs[rvim] - rhs[rvim]) / max(lhs[rvim], rhs[rvim])) ** 2
    dist += ((lhs[rdim] - rhs[rdim]) / max(lhs[rdim], rhs[rdim])) ** 2
    dist += ((lhs[rrdim] - rhs[rrdim]) / max(lhs[rrdim], rhs[rrdim])) ** 2
    dist += ((lhs[rfmim] - rhs[rfmim]) / max(lhs[rfmim], rhs[rfmim])) ** 2
    dist += ((lhs[rsmim] - rhs[rsmim]) / max(lhs[rsmim], rhs[rsmim])) ** 2
    dist = sqrt(dist)

    return dist


def save_list(list, filepath):
    file = open(filepath, "w")
    row = 1
    for sublist in list:
        file.write(str(row) + ": " + str(sublist) + "\n")
        row += 1


@print_timing
def text_to_proximity_list(text_filepath)
    canonical_features_dict = generate_alphabet_features(calculate_profiles=False, save=False)
    image = Image.open(text_filepath)
    symbol_coords = segment_line(image)
    symbol_images = images_from_coords(image, symbol_coords)
    proximity_list = []
    for image in symbol_images:
        features = calculate_features(image, calculate_profiles=False)
        symbol_proximity_list = []
        for key in canonical_features_dict:
            dist = get_distance(canonical_features_dict[key], features)
            symbol_proximity_list.append((key, 1 - dist))
        symbol_proximity_list.sort(reverse=True, key=lambda pair: pair[1])

        # Normalization of proximity.
        min_ = -1 * symbol_proximity_list[-1][1]
        symbol_proximity_list = [(key, (value + min_) / (1 + min_))
                                 for (key, value) in symbol_proximity_list]
        proximity_list.append(symbol_proximity_list)
    return proximity_list


def run():
    folder_path = "Data/lab6/"
    text_filepath = folder_path + "text.bmp"
    downsized_text_filepath = folder_path + "downsized_text.bmp"
    proximity_list = text_to_proximity_list(text_filepath)
    downsized_proximity_list = text_to_proximity_list(downsized_text_filepath)
    save_list(proximity_list, folder_path + "original_text_hypotheses.txt")
    save_list(downsized_proximity_list, folder_path + "downsized_text_hypotheses.txt")


run()

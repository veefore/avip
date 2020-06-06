from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
import csv
from utils import print_timing
from math import sqrt


# Image cropping.


def find_corners(image, val):
    # returns coordinates of the top left corner of a rectangle that contains all pixels that == val
    left = image.size[0]
    top = image.size[1]
    right = bot = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            if pixels[w, h] == val:
                left = min(left, w)
                top = min(top, h)
                right = max(right, w)
                bot = max(bot, h)
    return left, top, right, bot


def crop_image(image, val):
    # (0, 0)     top    (w - 1, 0)
    #
    # left               right
    #
    # (0, h - 1) bot (w - 1, h - 1)

    assert image.mode == '1'
    left, top, right, bot = find_corners(image, 0)
    return image.crop((left, top, right, bot))


# Image generation.


def text_to_image(text, font_path, font_size):
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = ImageDraw.Draw(Image.new('1', (1, 1))).textsize(text, font)
    image = Image.new('1', (text_width, text_height), 1)
    drawing = ImageDraw.Draw(image)
    drawing.text((0, 0), text, fill=0, font=font)
    return crop_image(image, 0)


@print_timing
def generate_symbol_images(text, font_path, font_size, folder_path):
    for symbol in text:
        text_to_image(symbol, font_path, font_size).save(folder_path + symbol + '.png')


# Calculations of the symbol features.


def get_weight(image):
    # calculates the weight of white
    weight = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            weight += pixels[w, h]
    return image.size[0] * image.size[1] - weight / 255


def get_centre_of_mass(image, weight=None):
    if weight is None:
        weight = get_weight(image)
    pixels = image.load()
    x = y = 0
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            # 255 - pixels[w, h] as it should be the weight of black, not white
            x += (255 - pixels[w, h]) * w
            y += (255 - pixels[w, h]) * h
    x /= weight * 255
    y /= weight * 255
    return int(x), int(y)


def get_relative_centre_of_mass(image, centre_of_mass=None):
    if centre_of_mass is None:
        centre_of_mass = get_centre_of_mass(image)
    centre_of_mass = ((centre_of_mass[0] - 1) / (image.size[0] - 1),
                      (centre_of_mass[1] - 1) / (image.size[1] - 1))
    return centre_of_mass


def get_inertia_moment_helper(image, p, q, centre_of_mass):
    if centre_of_mass is None:
        centre_of_mass = get_centre_of_mass(image)
    val = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            val += (p * (h - centre_of_mass[1]) + q * (w - centre_of_mass[0])) ** 2 * (255 - pixels[w, h])
    return val / 255


def get_inertia_moments(image, centre_of_mass=None):
    return get_inertia_moment_helper(image, 1, 0, centre_of_mass),\
           get_inertia_moment_helper(image, 0, 1, centre_of_mass)

def get_mixed_inertia_moment(image, centre_of_mass=None):
    if centre_of_mass is None:
        centre_of_mass = get_centre_of_mass(image)
    val = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            val += (w - centre_of_mass[0]) * (h - centre_of_mass[1]) * (255 - pixels[w, h])
    return val / 255


def get_diagonal_inertia_moments(image, centre_of_mass=None):
    return get_inertia_moment_helper(image, 1, -1, centre_of_mass) / 2,\
           get_inertia_moment_helper(image, 1, 1, centre_of_mass) / 2


def get_main_inertia_moments(image, horizontal_moment, vertical_moment):
    mixed_moment = get_mixed_inertia_moment(image)
    addendum1 = (horizontal_moment + vertical_moment) / 2
    addendum2 = sqrt((horizontal_moment - vertical_moment) ** 2 / 4 + mixed_moment ** 2)
    return addendum1 + addendum2, addendum1 - addendum2


def get_relative_inertia_moment(image, moment, weight=None):
    if weight is None:
        weight = get_weight(image)
    return moment / (weight ** 2)


def get_profiles(image):
    x_profile = np.zeros(image.size[0], dtype=int)
    y_profile = np.zeros(image.size[1], dtype=int)
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            x_profile[w] += 255 - pixels[w, h]
            y_profile[h] += 255 - pixels[w, h]
    return x_profile / 255, y_profile / 255


@print_timing
def calculate_features(image, calculate_profiles=True):
    w = 'weight'
    d = 'density'
    com = 'centre_of_mass'
    rcom = 'relative_centre_of_mass'
    him = 'horizontal_inertia_moment'
    vim = 'vertical_inertia_moment'
    rhim = 'relative_horizontal_inertia_moment'
    rvim = 'relative_vertical_inertia_moment'
    rdim = 'relative_diagonal_inertia_moment'
    rrdim = 'relative_reversed_diagonal_inertia_moment'
    rfmim = 'relative_first_main_inertia_moment'
    rsmim = 'relative_second_main_inertia_moment'

    dictionary = dict()
    dictionary[w] = get_weight(image)
    dictionary[d] = dictionary[w] / (image.size[0] * image.size[1])
    dictionary[com] = get_centre_of_mass(image)
    dictionary[rcom] = get_relative_centre_of_mass(image, dictionary[com])

    com = dictionary[com]
    weight = dictionary[w]
    dictionary['horizontal_inertia_moment'], dictionary['vertical_inertia_moment']\
        = get_inertia_moments(image, centre_of_mass=com)
    dictionary['relative_horizontal_inertia_moment']\
        = get_relative_inertia_moment(image, dictionary['horizontal_inertia_moment'], weight)
    dictionary['relative_vertical_inertia_moment']\
        = get_relative_inertia_moment(image, dictionary['vertical_inertia_moment'], weight)

    dictionary['diagonal_inertia_moment'], dictionary['reversed_diagonal_inertia_moment']\
        = get_diagonal_inertia_moments(image, centre_of_mass=com)
    dictionary['relative_diagonal_inertia_moment']\
        = get_relative_inertia_moment(image, dictionary['diagonal_inertia_moment'], weight)
    dictionary['relative_reversed_diagonal_inertia_moment']\
        = get_relative_inertia_moment(image, dictionary['reversed_diagonal_inertia_moment'], weight)

    dictionary['first_main_inertia_moment'], dictionary['second_main_inertia_moment']\
        = get_main_inertia_moments(image, dictionary['horizontal_inertia_moment'],
                                   dictionary['vertical_inertia_moment'])
    dictionary['relative_first_main_inertia_moment']\
        = get_relative_inertia_moment(image, dictionary['first_main_inertia_moment'], weight)
    dictionary['relative_second_main_inertia_moment']\
        = get_relative_inertia_moment(image, dictionary['second_main_inertia_moment'], weight)

    if calculate_profiles:
        horizontal_profile, vertical_profile = get_profiles(image)
        dictionary['horizontal_profile'] = horizontal_profile
        dictionary['vertical_profile'] = vertical_profile

    return dictionary


def save_features(dictionary, csv_file_path, name, profile_path):
    csv_file = open(csv_file_path, 'a')
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerow([name, dictionary['weight'], dictionary['density'],
                         dictionary['centre_of_mass'], dictionary['relative_centre_of_mass'],
                         dictionary['horizontal_inertia_moment'], dictionary['vertical_inertia_moment'],
                         dictionary['relative_horizontal_inertia_moment'], dictionary['relative_vertical_inertia_moment'],
                         dictionary['diagonal_inertia_moment'], dictionary['reversed_diagonal_inertia_moment'],
                         dictionary['relative_diagonal_inertia_moment'],
                         dictionary['relative_reversed_diagonal_inertia_moment'],
                         dictionary['first_main_inertia_moment'], dictionary['second_main_inertia_moment'],
                         dictionary['relative_first_main_inertia_moment'],
                         dictionary['relative_second_main_inertia_moment']])

    vertical_profile = dictionary['vertical_profile']
    plt.hist(x=[val for val in range(vertical_profile.size)],
             bins=vertical_profile.size, weights=vertical_profile)
    plt.savefig(profile_path + '_vertical_profile.png')
    plt.clf()

    horizontal_profile = dictionary['horizontal_profile']
    plt.hist(x=[val for val in range(horizontal_profile.size)],
             bins=horizontal_profile.size, weights=horizontal_profile,
             orientation=u'horizontal')
    plt.savefig(profile_path + '_horizontal_profile.png')
    plt.clf()


def features_from_text(text, folder_path, calculate_profiles=False):
    generate_symbol_images(text, "/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", 160, folder_path)
    features_dict = {}
    for symbol in text:
        image = Image.open(folder_path + symbol + '.png')
        features_dict[symbol] = calculate_features(image, calculate_profiles)
    return features_dict


def generate_alphabet_features(calculate_profiles=True, save=False):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    features_folder_path = 'Data/lab4/features/'
    features_dict = features_from_text(alphabet, folder_path='Data/lab4/alphabet/',
                                       calculate_profiles=calculate_profiles)
    if save:
        for symbol in alphabet:
            save_features(features_dict[symbol], features_folder_path + 'features.csv', symbol, features_folder_path + symbol)
    return features_dict

generate_alphabet_features()

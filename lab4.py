from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
from functools import wraps
import csv
from matplotlib import pyplot as plt


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


def find_top_left(image, val):
    # returns coordinates of the top left corner of a rectangle that contains all pixels that == val
    top = image.size[1]
    left = image.size[0]
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            if pixels[w, h] == val:
                top = min(top, h)
                left = min(left, w)
    return top, left


def find_bot_right(image, val):
    # returns coordinates of the bottom right corner of a rectangle that contains all pixels that == val
    bot = 0
    right = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            if pixels[w, h] == val:
                bot = max(bot, h)
                right = max(right, w)
    return bot, right


def crop_to_minimum(image):
    # assumes image.mode == '1'

    # (0, 0)     top    (w - 1, 0)
    #
    # left               right
    #
    # (0, h - 1) bot (w - 1, h - 1)

    top, left = find_top_left(image, 0)
    bot, right = find_bot_right(image, 0)
    width = right - left + 1
    height = bot - top + 1
    new_image = Image.new(image.mode, (width, height))
    new_pixels = new_image.load()
    pixels = image.load()
    for w in range(width):
        for h in range(height):
            new_pixels[w, h] = pixels[left + w, top + h]
    return new_image


def text_to_image(text, font_path, font_size):
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = ImageDraw.Draw(Image.new('1', (1, 1))).textsize(text, font)
    image = Image.new('1', (text_width, text_height), 1)
    drawing = ImageDraw.Draw(image)
    drawing.text((0, 0), text, fill=0, font=font)
    return crop_to_minimum(image)


@print_timing
def generate_symbol_images(alphabet, font_path, font_size):
    for symbol in alphabet:
        text_to_image(symbol, font_path, font_size).save('Data/lab4/alphabet/' + symbol + '.png')


def get_weight(image):
    # calculates the weight of white
    # assumes image.mode == '1'
    weight = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            weight += pixels[w, h]
    # weight of black for a '1' mode image is the area of the image minus the weight of white
    return image.size[0] * image.size[1] - weight / 255


def get_centre_of_mass(image, weight=-1):
    if weight == -1:
        weight = get_weight(image)
    pixels = image.load()
    x = 0
    y = 0
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            # 255 - pixels[w, h] as it should be the weight of black, not white
            x += (255 - pixels[w, h]) * w
            y += (255 - pixels[w, h]) * h
    x /= weight * 255
    y /= weight * 255
    return (int(x), int(y))


def get_relative_centre_of_mass(image, centre_of_mass=(-1, -1)):
    if centre_of_mass == (-1, -1):
        centre_of_mass = get_centre_of_mass(image)
    centre_of_mass = ((centre_of_mass[0] - 1) / (image.size[0] - 1),
                      (centre_of_mass[1] - 1) / (image.size[1] - 1))
    return centre_of_mass


def get_horizontal_inertia_moment(image, centre_of_mass=(-1, -1)):
    if centre_of_mass == (-1, -1):
        centre_of_mass = get_centre_of_mass(image)
    val = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            val += (h - centre_of_mass[1]) ** 2 * (255 - pixels[w, h])
    # / 255 as pixels[w, h] == 255 instead of 1 for the perfectly white pixel
    return val / 255


def get_vertical_inertia_moment(image, centre_of_mass=(-1, -1)):
    if centre_of_mass == (-1, -1):
        centre_of_mass = get_centre_of_mass(image)
    val = 0
    pixels = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            val += (w - centre_of_mass[0]) ** 2 * (255 - pixels[w, h])
    return val / 255


def get_relative_inertia_moment(image, moment):
    return moment / (image.size[0] ** 2 + image.size[1] ** 2)


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
def calculate_features(image, csv_file_path, name, profile_path_without_extension):
    weight = get_weight(image)
    density = weight / (image.size[0] * image.size[1])
    centre_of_mass = get_centre_of_mass(image)
    relative_centre_of_mass = get_relative_centre_of_mass(image, centre_of_mass)
    horizontal_inertia_moment = get_horizontal_inertia_moment(image, centre_of_mass)
    vertical_inertia_moment = get_vertical_inertia_moment(image, centre_of_mass)
    relative_horizontal_inertia_moment = get_relative_inertia_moment(image, horizontal_inertia_moment)
    relative_vertical_inertia_moment = get_relative_inertia_moment(image, vertical_inertia_moment)
    horizontal_profile, vertical_profile = get_profiles(image)

    #print('Weight ', weight)
    #print('Density ', density)
    #print('Center of mass ', centre_of_mass)
    #print('Relative centre of mass ', relative_centre_of_mass)
    #print('Horizontal inertia moment ', horizontal_inertia_moment)
    #print('Vertical inertia moment ', vertical_inertia_moment)
    #print('Relative horizontal inertia moment ', relative_horizontal_inertia_moment)
    #print('Relative vertical inertia moment ', relative_vertical_inertia_moment)

    csv_file = open(csv_file_path, 'a')
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerow([name, weight, density, centre_of_mass, relative_centre_of_mass, horizontal_inertia_moment,\
                         vertical_inertia_moment, relative_horizontal_inertia_moment, relative_vertical_inertia_moment])

    plt.hist(x=[val for val in range(vertical_profile.size)],
             bins=vertical_profile.size, weights=vertical_profile)
    plt.savefig(profile_path_without_extension + '_vertical_profile.png')
    plt.clf()


    plt.hist(x=[val for val in range(horizontal_profile.size)],
             bins=horizontal_profile.size, weights=horizontal_profile,
             orientation=u'horizontal')
    plt.savefig(profile_path_without_extension + '_horizontal_profile.png')
    plt.clf()


def run():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    generate_symbol_images(alphabet, "/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", 80)
    for symbol in alphabet:
        image = Image.open('Data/lab4/alphabet/' + symbol + '.png')
        calculate_features(image, 'Data/lab4/features/features.csv', symbol, 'Data/lab4/features/' + symbol)

#run()

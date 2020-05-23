from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
from functools import wraps
import operator
from lab4 import print_timing, get_profiles, crop_to_minimum



@print_timing
def segment_profile(profile, coef=1):
    mean = np.mean(profile) * coef
    marks = []
    start = None
    for i in range(profile.size):
        if profile[i] > mean:
            if start is None:
                start = i
        elif profile[i] < mean:
            if start is not None:
                marks.append((start, i))
                start = None
    if start is not None:
        marks.append((start, profile.size - 1))
    return marks


def segment_symbol_profile(profile, coef=1):
    mean = np.mean(profile) * coef
    start = None
    end = None
    for i in range(profile.size):
        if profile[i] > mean and start is None:
            start = i
        elif profile[i] <= mean and profile[i - 1] > mean:
            end = i
    if profile[profile.size - 1] > mean:
        end = profile.size - 1
    return start, end



def segment_symbol(image):
    horizontal_profile, vertical_profile = get_profiles(image)
    horizontal_mark = segment_symbol_profile(horizontal_profile, 0.1)
    vertical_mark = segment_symbol_profile(vertical_profile, 0.1)
    return (horizontal_mark[0], vertical_mark[0], horizontal_mark[1], vertical_mark[1])


@print_timing
def segment_line(image):
    horizontal_profile, vertical_profile = get_profiles(image)
    horizontal_marks = segment_profile(horizontal_profile, 0.1)
    vertical_marks = segment_profile(vertical_profile, 0.1)
    #print(horizontal_marks)
    #print(vertical_marks)
    symbols = []
    itr = 0
    for x_start, x_end in horizontal_marks:
        for y_start, y_end in vertical_marks:
            crop_window = (x_start, y_start, x_end, y_end)
            symbol = image.crop(crop_window) # Crops according to line profile
            symbol = symbol.crop(segment_symbol(symbol)) # Crop according to symbol profile
            symbol.save("Data/lab5/" + str(itr) + ".png")
            symbols.append((x_start, y_start, x_end, y_end))
            itr = itr + 1
    return symbols


def run():
    filepath = "Data/lab5/text.bmp"
    image = Image.open(filepath)
    segment_line(image)


run()

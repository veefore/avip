from PIL import Image
import numpy as np
from utils import print_timing
from lab4 import get_profiles


@print_timing
def segment_profile(profile, threshold):
    marks = []
    start = None
    for i in range(profile.size):
        # If the interval start hasn't yet been set and the current value is above the threshold, then set it.
        if profile[i] > threshold and start is None:
            start = i
        # If the interval start has been set and the current value is below the threshold, then set the interval end.
        elif profile[i] <= threshold and start is not None:
            marks.append((start, i))
            start = None
    if start is not None:
        marks.append((start, profile.size - 1))
    return marks


def segment_symbol_profile(profile, threshold):
    start = None
    end = None
    for i in range(profile.size):
        if profile[i] > threshold and start is None:
            start = i
        elif profile[i] <= threshold < profile[i - 1]:
            end = i
    if profile[profile.size - 1] > threshold:
        end = profile.size - 1
    return start, end


def segment_symbol(image):
    horizontal_profile, vertical_profile = get_profiles(image)
    horizontal_mark = segment_symbol_profile(horizontal_profile, 0.1 * np.mean(horizontal_profile))
    vertical_mark = segment_symbol_profile(vertical_profile, 0.1 * np.mean(vertical_profile))
    return horizontal_mark[0], vertical_mark[0], horizontal_mark[1], vertical_mark[1]


@print_timing
def segment_line(image, folder_path=None):
    horizontal_profile, vertical_profile = get_profiles(image)
    horizontal_marks = segment_profile(horizontal_profile, 0.1 * np.mean(horizontal_profile))
    vertical_marks = segment_profile(vertical_profile, 0.1 * np.mean(vertical_profile))
    symbols = []
    itr = 0
    for x_start, x_end in horizontal_marks:
        for y_start, y_end in vertical_marks:
            crop_window = (x_start, y_start, x_end, y_end)
            symbol = image.crop(crop_window)  # Crops according to line profile.
            symbol = symbol.crop(segment_symbol(symbol))  # Crop according to symbol profile.
            if folder_path is not None:
                symbol.save(folder_path + str(itr) + ".png")
                itr = itr + 1
            symbols.append((x_start, y_start, x_end, y_end))
    return symbols


@print_timing
def run(folder_path):
    file_path = folder_path + "text.bmp"
    image = Image.open(file_path)
    segment_line(image, folder_path)


#run("Data/lab5/")

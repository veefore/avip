from PIL import Image
import math
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


def multiply_tuple(tup, coef):
    return tuple([math.floor(x * coef) for x in list(tup)])


@print_timing
def upsample(image, coef):
    # isinstance(coef, int) is asserted
    pixels = image.load()
    new_image = Image.new('RGB', multiply_tuple(image.size, coef))
    new_pixels = new_image.load()
    for h in range(image.size[0]):
        for w in range(image.size[1]):
            for shift_h in range(coef):
                for shift_w in range(coef):
                    new_pixels[h * coef + shift_h, w * coef + shift_w] = pixels[h, w]
    return new_image


@print_timing
def downsample(image, coef):
    # isinstance(coef, int) is asserted
    pixels = image.load()
    new_image = Image.new('RGB', multiply_tuple(image.size, 1 / coef))
    new_pixels = new_image.load()
    for h in range(new_image.size[0]):
        for w in range(new_image.size[1]):
            new_pixels[h, w] = pixels[h * coef, w * coef]
    return new_image


@print_timing
def resample_2pass(image, n, m):
    # image = image * (n / m)
    return downsample(upsample(image, n), m)


@print_timing
def resample_1pass(image, n, m):
    # image = image * (n / m)
    pixels = image.load()
    new_size = multiply_tuple(image.size, n / m)
    new_image = Image.new('RGB', new_size)
    new_pixels = new_image.load()
    for h in range(new_size[0]):
        for w in range(new_size[1]):
            new_pixels[h, w] = pixels[math.floor(h * m / n), math.floor(w * m / n)]
    return new_image


@print_timing
def halftone(image):
    pixels = image.load()
    new_image = Image.new('L', image.size)
    new_pixels = new_image.load()
    for h in range(image.size[0]):
        for w in range(image.size[1]):
            new_pixels[h, w] = (round((pixels[h, w][0] + pixels[h, w][1] + pixels[h, w][2]) / 3))
    return new_image


@print_timing
def Otsu_binarization(image):
    # image.mode == 'L' is asserted
    hist = [0] * 256

    pixels = image.load()
    # fill the hist
    for h in range(image.size[0]):
        for w in range(image.size[1]):
            hist[pixels[h, w]] = hist[pixels[h, w]] + 1

    w0 = 0
    w1 = 255
    max_rel = -1
    corr_t = -1
    # iterate over all thresholds
    for t in range(256):
        # update relative frequencies
        w0 += hist[t]
        w1 -= hist[t]

        # calculate average brightnesses
        m0 = 0
        m1 = 0
        for p in hist:
            if p <= t:
                m0 += round(hist.index(p) * p / w0)
            else:
                m1 += round(hist.index(p) * p / w1)

        # calculate variances
        d0 = 0
        d1 = 0
        for p in hist:
            if p <= t:
                d0 += p * (hist.index(p) - m0) ** 2
            else:
                d1 += p * (hist.index(p) - m1) ** 2

        # maximize the relation of between-group variance to within-group variance
        d_wg = w0 * d0 + w1 * d1
        d_bg = w0 * w1 * (m0 - m1) ** 2
        rel = d_bg / d_wg
        if max_rel == -1:
            max_rel = rel
            corr_t = t
        else:
            if max_rel < rel:
                max_rel = rel
                corr_t = t

    new_image = Image.new('1', image.size)
    new_pixels = new_image.load()
    for h in range(image.size[0]):
        for w in range(image.size[1]):
            if pixels[h, w] > corr_t:
                new_pixels[h, w] = (1)
            else:
                new_pixels[h, w] = (0)

    return new_image


def run_test():
    img1 = Image.open("Data/lab1/img1.bmp")
    upsample(img1, 3).save("Data/lab1/upsampled_img1.bmp")
    downsample(img1, 3).save("Data/lab1/downsampled_img1.bmp")
    resample_2pass(img1, 7, 3).save("Data/lab1/2pass_resampled_img1.bmp")
    resample_1pass(img1, 7, 3).save("Data/lab1/1pass_resampled_img1.bmp")
    halftone(img1).save("Data/lab1/halftone_img1.bmp")
    Otsu_binarization(halftone(img1)).save("Data/lab1/binarized_img1.bmp")

    img2 = Image.open("Data/lab1/img2.bmp")
    Otsu_binarization(halftone(img2)).save("Data/lab1/Otsu_binarized_img2.bmp")
    img3 = Image.open("Data/lab1/img3.bmp")
    Otsu_binarization(halftone(img3)).save("Data/lab1/Otsu_binarized_img3.bmp")
    img4 = Image.open("Data/lab1/img4.bmp")
    Otsu_binarization(halftone(img4)).save("Data/lab1/Otsu_binarized_img4.bmp")

run_test()

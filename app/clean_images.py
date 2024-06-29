# Contains functions which resizes images and cleans them

import os

from PIL import Image

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def clean_images(path_to_extract = "Datasets/images/", path_to_save = "Datasets/cleaned_images/", image_size = 64):
    dirs = os.listdir(path_to_extract)
    final_size = image_size
    for n, item in enumerate(dirs, 1):

        im = Image.open(path_to_extract + item)

        new_im = resize_image(final_size, im)

        new_im.save(path_to_save + item)

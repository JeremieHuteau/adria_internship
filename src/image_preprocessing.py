import argparse
import os
import pickle
import math
import multiprocessing

from PIL import Image

def main(args):
    image_names = filter(
            lambda name: os.path.isfile(os.path.join(args.input_dir, name)),
            os.listdir(args.input_dir))

    image_paths = map(
            lambda name: os.path.join(args.input_dir, name),
            image_names)

    target_size = (args.width, args.height)

    resize_images(image_paths, target_size, args.output_dir,
            args.num_cpus)

def resize_images(image_paths, target_size, output_dir, num_workers=4):
    map_args = map(
            lambda image_path: (image_path, target_size, output_dir),
            image_paths)

    with multiprocessing.Pool(num_workers) as pool:
        for _ in pool.imap_unordered(
                resize_task, map_args, chunksize=8):
            pass


def resize_task(args):
    resize_save(*args)

def resize_save(image_path, target_size, output_dir):
    image = Image.open(image_path)

    image = resize_keep_aspect(image, target_size)

    image.save(os.path.join(output_dir, os.path.basename(image_path)))

def resize_keep_aspect(image, size, resample=Image.BICUBIC, reducing_gap=3.0):
    # PIL.Image.thumbnail modified 
    # Upscaling if image is smaller than size,
    # resizes image such that smallest (and not largest as originally done in
    # thumbnail) dimension is equal to size.
    # ('##' comments document the modifications)

    x, y = map(math.floor, size)
    
    ## Enables upscaling
    #if x >= image.width and y >= image.height:
    #    return

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    # preserve aspect ratio
    aspect = image.width / image.height
    ## if/else are switched to have the smallest dimension equal to size
    if x / y >= aspect:
        y = round_aspect(
            x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    else:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    size = (x, y)

    box = None
    if reducing_gap is not None:
        res = image.draft(None, (size[0] * reducing_gap, size[1] * reducing_gap))
        if res is not None:
            box = res[1]

    if image.size != size:
        im = image.resize(size, resample, box=box, reducing_gap=reducing_gap)

        image.im = im.im
        image._size = size
        image.mode = image.im.mode

    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Resize (preserving aspect ratio) all images"
                " in input_dir and save them in output_dir")

    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--width', type=int, default=256,
            help="minimum width of resized images (default 256)")
    parser.add_argument('--height', type=int, default=256,
            help="minimum height of resized images (default 256)")
    parser.add_argument('--num-cpus', type=int,
            default=4)

    args = parser.parse_args()
    main(args)


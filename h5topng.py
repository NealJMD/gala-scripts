import numpy as np
import scipy.misc
import h5py
import random
import sys
import argparse

def main(h5_filename, output_dir, color_style, key, im_limit=1000, sideways=False):
    image_stack = read_h5_image_stack(h5_filename, key)
    if sideways:image_stack = np.transpose(image_stack,(1,0,2))
    if im_limit < image_stack.shape[0]: image_stack = image_stack[:im_limit,:,:]
    color_map = build_color_map(image_stack, color_style)
    output_pngs(image_stack, color_map, output_dir)

def output_pngs(image_stack, color_map, output_dir, channel_id=""):
    print "Handling image_stack with shape %s" % (str(image_stack.shape))
    if image_stack.ndim > 3:
        for jj in range(image_stack.shape[image_stack.ndim-1]):
            output_pngs(image_stack[..., jj], color_map, output_dir, "-"+str(jj))
        return
    print "Printing pngs to %s" % (output_dir)
    (image_count, width, height) = image_stack.shape
    for ii in xrange(image_count):
        labels = image_stack[ii, ...]
        output = np.zeros((width, height, 3))
        for w in xrange(width):
            for h in xrange(height):
                colors = color_map[labels[w,h]]
                for channel, intensity in enumerate(colors):
                    output[w, h, channel] = intensity
        save_png(output, str(ii)+channel_id, output_dir)

def save_png(image_matrix, identifier, output_dir):
    filename = output_dir + str(identifier) + ".png"
    scipy.misc.imsave(filename, image_matrix)

def read_h5_image_stack(h5_filename, key):
    f = h5py.File(h5_filename, "r")
    return np.array(f[key])

def build_color_map(image_stack, color_style):
    if color_style == "random": return build_random_color_map(image_stack)
    elif color_style == "grayscale": return build_grayscale_color_map(image_stack)
    else: raise NameError("color_style must be 'random' or 'grayscale'")

def build_grayscale_color_map(image_stack):
    color_map = {}
    uniques = np.unique(image_stack)
    max_val = float(np.max(image_stack))
    for pixel_value in uniques:
        luminance = int(255 * pixel_value / max_val)
        color_map[pixel_value] = [luminance, luminance, luminance]
    return color_map

def build_random_color_map(image_stack):
    color_map = {}
    uniques = np.unique(image_stack)
    for label in uniques:
        color_map[label] = pick_random_color()
    return color_map

def pick_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return [r,g,b]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=str,
                    help="Path to the target h5")
    parser.add_argument("out_path", type=str,
                    help="Path and prefix of pngs")
    parser.add_argument("--side-view", action="store_true",
                    help="output lateral cross-sections instead of frontals")
    parser.add_argument("--random-colors", action="store_true",
                    help="output with radnom colormap")
    parser.add_argument("--im-limit", type=int, default=1000,
                    help="limit of images to output")
    parser.add_argument("--h5-key", type=str, default="stack",
                    help="path to")
    args = parser.parse_args()
    if args.random_colors: color = "random"
    else: color="grayscale"
    main(args.h5_path, args.out_path, color, args.h5_key, im_limit=args.im_limit, sideways=args.side_view)



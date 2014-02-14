import numpy as np
import scipy.misc
import h5py
import random
import sys

def main(h5_filename, output_dir, color_style="random", key="stack"):
	image_stack = read_h5_image_stack(h5_filename, key)
	color_map = build_color_map(image_stack, color_style)
	output_pngs(image_stack, color_map, output_dir)

def output_pngs(image_stack, color_map, output_dir):
	(image_count, width, height) = image_stack.shape
	for ii in xrange(image_count):
		labels = image_stack[ii, ...]
		output = np.zeros((width, height, 3))
		for w in xrange(width):
			for h in xrange(height):
				colors = color_map[labels[w, h]]
				for channel, intensity in enumerate(colors):
					output[w, h, channel] = intensity
		save_png(output, ii, output_dir)

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
	if len(sys.argv) < 3: print "Usage: python h5topng.py <h5 path> <output path> [grayscale|random] [h5 key]"
	elif len(sys.argv) < 4:	main(sys.argv[1], sys.argv[2])
        elif len(sys.argv) < 5: main(sys.argv[1], sys.argv[2], sys.argv[3])
        else: main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

import libtiff
from libtiff import TIFF
import numpy as np
import h5py
import os, sys


def slice_3d_array(iterable, row_end, col_end, frame_end, 
                row_start=0, col_start=0, frame_start=0):
    """
    Parameters
    ----------
    iterable: an np array or another iterable (eg tmy_tif.iter_images)
    """
    row_end = int(row_end)
    col_end = int(col_end)
    frame_end = int(frame_end)
    row_start = int(row_start)
    col_start = int(col_start)
    frame_start = int(frame_start)
    rows = row_end - row_start
    cols = col_end-col_start
    frames = frame_end-frame_start

    specified_stack = np.zeros((frames, rows, cols))
    stacked_image_index = frame_start

    if hasattr(iterable, '__call__'):
        enumerable = enumerate(iterable())
    else:
        enumerable = enumerate(iterable)
    for index, image in enumerable:

        if stacked_image_index >= frame_end: break

        np_image = np.array(image)
        total_rows, total_cols = np_image.shape
        np_specified = np_image[row_start:row_end, col_start:col_end]
        specified_stack[stacked_image_index, :, :] = np_specified
        stacked_image_index += 1
    return specified_stack


def slice_and_convert(input_filename, output_filename, 
    normalization=1, input_stack_path="", output_groups=[], 
    output_stack_label="", slice_3d_args=[]):

    name_in, extension_in = os.path.splitext(input_filename)
    name_out, extension_out = os.path.splitext(output_filename)

    if extension_in.lower() in [".tif", ".tiff"]:
        iterable = read_tif(input_filename)
    elif extension_in.lower() in [".h5"]:
        iterable = read_h5(input_filename, input_stack_path)
    else:
        raise FileError("%s is not a tif nor an h5!" % (input_filename))

    stack = slice_3d_array(iterable, *slice_3d_args) / float(normalization)

    if extension_out.lower() in [".tif", ".tiff"]:
        write_tif(stack, output_filename)
    elif extension_out.lower() in [".h5"]:
        write_h5(stack, output_filename, output_groups, output_stack_label)
    else:
        raise FileError("%s in not a tif nor an h5!" % (output_filename))

def read_tif(tif_filename):
    tif = TIFF.open(tif_filename, mode='r')
    iterable = tif.iter_images
    return iterable

def write_tif(stack, output_filename):
    tif = TIFF.open(output_filename, mode='w')
    tif.write_image(stack)

def read_h5(h5_filename, input_stack_path):
    f = h5py.File(h5_filename,'r')
    dset = f[input_stack_path]
    return dset

def write_h5(stack, output_filename, groups, stack_label):
    f = h5py.File(output_filename, 'w')
    current_group = f
    for group in groups:
        current_group = current_group.create_group(group)
    current_group.create_dataset(stack_label, data=stack, compression='lzf')
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3: 
        raise Exception( "Usage: python sliceconvert.py input_file output_file rows cols frames [row__start col_start frame_start]")
    slice_3d_args = tuple(sys.argv[3:])
    og = ["volume"]
#    og = []
    osl = "predictions"
#    osl = "stack"
    slice_and_convert(sys.argv[1], sys.argv[2], normalization=255.0, 
        input_stack_path="stack", output_groups=og, 
        output_stack_label=osl, slice_3d_args=slice_3d_args)


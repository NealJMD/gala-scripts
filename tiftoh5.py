import libtiff
from libtiff import TIFF
import numpy as np
import h5py
import sys


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
    for index, image in enumerate(iterable()):

        if stacked_image_index >= frame_end: break

        np_image = np.array(image)
        total_rows, total_cols = np_image.shape
        np_specified = np_image[row_start:row_end, col_start:col_end]
        specified_stack[stacked_image_index, :, :] = np_specified
        stacked_image_index += 1
    return specified_stack


def tif_slice_to_h5(tif_filename, output_filename, normalization, groups, stack_label, *slice_3d_args):
    tif = TIFF.open(tif_filename, mode='r')
    iterable = tif.iter_images
    stack = slice_3d_array(iterable, *slice_3d_args) / normalization
    f = h5py.File(output_filename, 'w')
    current_group = f
    for group in groups:
        current_group = current_group.create_group(group)
    current_group.create_dataset(stack_label, data=stack, compression='lzf')
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 3: 
        raise Exception( "Usage: python tiftoh5.py input_tif output_tif rows cols frames [row__start col_start frame_start]")
    slice_3d_args = tuple(sys.argv[3:])
    tif_slice_to_h5(sys.argv[1], sys.argv[2], 255.0, ["volume"], "predictions", *slice_3d_args)


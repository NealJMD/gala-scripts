import libtiff
from libtiff import TIFF
import numpy as np
import h5py
import os, sys
from gala.morpho import relabel_connected

def ensure_path(path):
    if os.path.exists(path): return path
    os.makedirs(path)
    return path

def ensure_file(path):
    head, tail = os.path.split(path)
    ensure_path(head)
    return path

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
    is_ints = True

    if hasattr(iterable, '__call__'):
        enumerable = enumerate(iterable())
    else:
        enumerable = enumerate(iterable)
    for index, image in enumerable:

        if stacked_image_index >= frame_end: break
        np_image = np.array(image)
        if not issubclass(type(np_image[0,0]), np.integer): is_ints = False 
        total_rows, total_cols = np_image.shape
        np_specified = np_image[row_start:row_end, col_start:col_end]
        specified_stack[stacked_image_index, :, :] = np_specified
        stacked_image_index += 1
    if is_ints: return specified_stack.astype(int)
    return specified_stack

def add_2d_derivative_channel(original_stack, concat=True):
    print "Creating derivative channel from stack of shape %s" % (str(original_stack.shape))
    stack = np.zeros_like(original_stack)
    for ii, frame in enumerate(original_stack):
        grads = np.gradient(frame)
        for grad in grads:
            stack[ii, ...] += (grad * grad)
    stack = np.sqrt(stack)
    if not concat: return stack
    return np.concatenate((original_stack[...,np.newaxis], stack[...,np.newaxis]), axis=3)
         

def slice_and_convert(input_filename, output_filename, 
    normalization=1, input_stack_path="", output_groups=[], 
    output_stack_label="", postprocess="", slice_3d_args=[]):

    name_in, extension_in = os.path.splitext(input_filename)
    name_out, extension_out = os.path.splitext(output_filename)

    ensure_file(output_filename)
    if extension_in.lower() in [".tif", ".tiff"]:
        iterable = read_tif(input_filename)
    elif extension_in.lower() in [".h5"]:
        iterable = read_h5(input_filename, input_stack_path)
    else:
        raise Exception("%s is not a tif nor an h5!" % (input_filename))

    print "Read in %s, slicing with args: %s" % (input_filename, str(slice_3d_args))
    stack = slice_3d_array(iterable, *slice_3d_args) / normalization
    print "Cut down to size %s" % (str(stack.shape))
    if postprocess == "relabel":
        print "Ensuring contiguous labels..."
        stack = relabel_connected(stack)
    elif postprocess == "derivative":
        print "Adding derivative channel...",
        stack = add_2d_derivative_channel(stack, concat=True)
        print "new shape: %s" % (str(stack.shape))

    print "Writing out to %s" % (output_filename)
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

def main():
    usage = "Usage: python sliceconvert.py <contlabels|labels|values|ders> input_file output_file rows cols frames [row__start col_start frame_start]"
    if len(sys.argv) < 3:
        print usage
        return
    slice_3d_args = tuple(sys.argv[4:])
    postprocess = ""
    if sys.argv[1] == "labels" or sys.argv[1] == "contlabels":
        out_group = []
        out_stack_label = "stack"
        in_path = out_stack_label
        norm = 1
        if sys.argv[1] == "contlabels":  postprocess="relabel"
    elif sys.argv[1] == "values" or sys.argv[1] == "ders":
        out_group = ["volume"]
        out_stack_label = "predictions"
        in_path = "volume/predictions"
        #out_group = []
        #out_stack_label = "stack"
        #in_path = "stack"
        norm = 255.0
        if sys.argv[1] == "ders": postprocess = "derivative"
    else:
        print usage
        return
    slice_and_convert(sys.argv[2], sys.argv[3], normalization=norm, 
        input_stack_path=in_path, output_groups=out_group,
        output_stack_label=out_stack_label, postprocess=postprocess,
        slice_3d_args=slice_3d_args)

if __name__ == "__main__":
    main()

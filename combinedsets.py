from rungala import get_paths
from sliceconvert import add_2d_derivative_channel
from sliceconvert import ensure_file
import argparse
import gala.imio as imio
import numpy as np

ALL_VOLUME_IDS = ["topleft", "topright", "bottomleft", "bottomright"]
ID_DELIMITER = "AND"
DERIVATIVE_ID = "der"

def write_h5(arr, path, group):
    print "Writing stack with shape %s to %s" % (str(arr.shape), path)
    ensure_file(path)
    imio.write_h5_stack(arr, path, group=group, compression="lzf")
    

def add_gradient_channel(traintest, size, volume_id, cues_id, datatype, group):
    last_component = cues_id.split(ID_DELIMITER)[-1]
    new_cues_id = cues_id+ID_DELIMITER+DERIVATIVE_ID+last_component
    src_paths = get_paths("XX", traintest, size, volume_id, cues_id, "XX", "XX", "XX")
    dest_paths = get_paths("XX", traintest, size, volume_id, new_cues_id, "XX", "XX", "XX")
    src_arr = imio.read_h5_stack(src_paths[datatype], group=group)
    dest_arr = add_2d_derivative_channel(src_arr, concat=True)
    write_h5(dest_arr, dest_paths[datatype], group)
   

def concatenate(traintest, size, volume_id, cues_id_1, cues_id_2, datatype, group):
    cues_id_new = cues_id_1+ID_DELIMITER+cues_id_2
    src_paths_1 = get_paths("XX", traintest, size, volume_id, cues_id_1, "XX", "XX", "XX")
    src_paths_2 = get_paths("XX", traintest, size, volume_id, cues_id_2, "XX", "XX", "XX")
    dest_paths = get_paths("XX", traintest, size, volume_id, cues_id_new, "XX", "XX", "XX")
    src_arr_1 = imio.read_h5_stack(src_paths_1[datatype], group=group)
    src_arr_2 = imio.read_h5_stack(src_paths_2[datatype], group=group)
    if src_arr_1.ndim < 4: src_arr_1 = src_arr_1[...,np.newaxis]
    if src_arr_2.ndim < 4: src_arr_2 = src_arr_2[...,np.newaxis]
    dest_arr = np.concatenate((src_arr_1, src_arr_2), axis=3)
    write_h5(dest_arr, dest_paths[datatype], group)


def main():
    # traintest, size, volume_id, cues_id
    parser = argparse.ArgumentParser()
    parser.add_argument("traintest", type=str)
    parser.add_argument("size", type=str)
    parser.add_argument("volume_id", type=str)
    parser.add_argument("cues_id", type=str)
    parser.add_argument("--concat", type=str, default="",
                    help="cues_id of target to concatenate")
    parser.add_argument("--group", type=str, default="volume/predictions")
    parser.add_argument("--datatype", type=str, default="hypercubes")
    parser.add_argument("--gradient", action="store_true",
                    help="whether to add the gradient")
    args = parser.parse_args()
    if args.volume_id == "all":
        for new_vol_id in ALL_VOLUME_IDS:
            args.volume_id = new_vol_id
            multiplex(args)
    else: multiplex(args)

def multiplex(a):
    if a.gradient:
        add_gradient_channel(a.traintest, a.size, a.volume_id, a.cues_id, a.datatype, a.group)
    elif len(a.concat) > 0:
        concatenate(a.traintest, a.size, a.volume_id, a.cues_id, a.concat, a.datatype, a.group)


if __name__ == "__main__":
    main()

from rungala import get_paths
import argparse
import numpy as np
import h5py
import tifffile
import os

H5_SUFFIX = "-0.50.lzf.h5"
TIF_SUFFIX = "-0.50.tif"
H5_KEY = "stack"
ID_DELIMITER = "AND"

def format(cues_id, features_id, exec_id, outfile_name):
    paths = get_paths("gala-segment", "test", "whole", "only", cues_id, features_id, exec_id, "only")
    seg_path = os.path.join(paths["output_dir"], paths["experiment_name"]+H5_SUFFIX)
    if len(outfile_name) < 1:
        out_path = os.path.join(paths["output_dir"], paths["experiment_name"]+TIF_SUFFIX)
    else:
        out_path = os.path.join(paths["output_dir"], outfile_name)
    seg_h5 = h5py.File(seg_path, "r")
    seg_np = np.array(seg_h5[H5_KEY]).astype(np.int16)
    print "saving to",out_path
    tifffile.imsave(out_path, seg_np)

def main():
    # traintest, size, volume_id, cues_id
    parser = argparse.ArgumentParser()
    parser.add_argument("cues_id", type=str)
    parser.add_argument("features_id", type=str)

    parser.add_argument("--outfile-name", type=str, default="",
                    help="special id to target")
    parser.add_argument("--exec-id", type=str, default="",
                    help="special id to target")
    args = parser.parse_args()
    format(args.cues_id, args.features_id, args.exec_id, args.outfile_name)

if __name__ == "__main__":
    main()

from gala import morpho, imio, agglo, evaluate
import numpy as np
from rungala import get_specified_file_path, get_paths
from sliceconvert import ensure_file
import argparse
import os
import h5topng
import h5py


fractions = [(0, 128, 0, 64, 0, 32, "topleft", "128x64x32"),
             (0, 512, 0, 512, 0, 100, "topleft", "quarter"),
             (0, 512, 512, 1024, 0, 100, "topright", "quarter"),
             (512, 1024, 0, 512, 0, 100, "bottomleft", "quarter"),
             (512, 1024, 512, 1024, 0, 100, "bottomright", "quarter")]

def read_h5(h5_filename, key):
    print "Read stack from",h5_filename,
    f = h5py.File(h5_filename, "r")
    stack = np.array(f[key])
    print "with shape",stack.shape,"and dtype",stack.dtype
    return stack

def write_h5(arr, path, group="stack", print_top_image=True, dry=False):
    arr += 1
    print "Writing stack with shape %s to %s" % (str(arr.shape), path)
    ensure_file(path)
    if not dry:
        imio.write_h5_stack(arr, path, group=group, compression="lzf")
    if not print_top_image: return
    top_im = arr[0,:,:][np.newaxis,:,:]
    colors = h5topng.build_color_map(top_im,"random")
    h5topng.output_pngs(top_im, colors, path+"-top-")

def invert_gray(im):
    return (im*-1)+1

def main(args):
    ws_id = generate_ws_id(args.min_seed_size, args.connectivity, 
        args.smooth_thresh, args.keep_speckles, args.override_skimage)
    paths = get_paths("XX", args.traintest, args.size, args.volume_id, 
        "idsia", "XX", "XX", "XX", ws_id)
    stack = read_h5(paths["hypercubes"], args.h5_key)
    if args.dry_run: stack = stack[:2,:,:]
    if not args.dont_invert: stack[...] = invert_gray(stack)
    ws = np.zeros(stack.shape)
    cur_max = 0
    for ii in range(stack.shape[0]):
        print ii
        ws[ii,:,:] = morpho.watershed(stack[ii,:,:], minimum_seed_size=args.min_seed_size,
            connectivity=args.connectivity, smooth_thresh=args.smooth_thresh,
            override_skimage=args.override_skimage) + cur_max
        cur_max = ws[ii,:,:].max()
    ws = ws.astype('int64')
    print "unique labels in ws:",np.unique(ws).size
    if not args.keep_speckles:
        ws = agglo.despeckle_watershed(ws)
        print "unique labels after despeckling:",np.unique(ws).size
    ws, _, _ = evaluate.relabel_from_one(ws)
    if ws.min() < 1: ws += (1-ws.min())
    write_h5(ws, paths["watersheds"], dry=args.dry_run)
    if args.no_quarters or (args.size != "whole"): return
    for row_start, row_end, col_start, col_end, frame_start, frame_end, label, size in fractions:
        path = generate_path(args.traintest, size, label, ws_id)
        write_h5(ws[frame_start:frame_end, row_start:row_end, col_start:col_end], path)

def generate_ws_id(minimum_seed_size, connectivity, smooth_thresh, keep_speckles, override, prob_thresh=1):
    if override: pre = "o"
    else: pre = ""
    if not keep_speckles: pre += "c"
    return "%smss%dcn%dst%dpr%d" % (pre, minimum_seed_size, connectivity, 
                                    int(smooth_thresh*100), prob_thresh)

def generate_path(traintest, size, volume_id, ws_id):
    specifier = ["watersheds", traintest, size, volume_id, ws_id]
    return get_specified_file_path(specifier,".lzf.h5")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("traintest", type=str)
    parser.add_argument("size", type=str)
    parser.add_argument("volume_id", type=str)
    parser.add_argument("--no-quarters", action="store_true",
        help="by default, any 'whole' volumes also save the quarters")
    parser.add_argument("--keep-speckles", action="store_true")
    parser.add_argument("--dry-run", action="store_true") 
    parser.add_argument("--out-path", type=str, default="")
    parser.add_argument("--h5-key", type=str, default="volume/predictions")
    parser.add_argument("--min-seed-size", type=int, default=1)
    parser.add_argument("--connectivity", type=int, default=2)
    parser.add_argument("--smooth-thresh", type=float, default=0.0)
    parser.add_argument("--override-skimage", action="store_true", default=False,
        help="use the pure python implementation for watershed")
    parser.add_argument("--dont-invert", action="store_true", default=False,
        help="by default, we invert the black and white scale")
    args = parser.parse_args()
    main(args)

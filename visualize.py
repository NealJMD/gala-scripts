import mayavi.mlab
import gala.viz as viz
import gala.imio as imio
import gala.evaluate as evaluate
import argparse
import numpy as np
from scipy.misc import imshow

LIST_CAP = 10

def plot_vi_breakdown(seg, gt):
    viz.plot_vi_breakdown(seg, gt, hlines=10)

def show_greatest_vi(seg, gt):
    (worst_false_merges, merge_ents, worst_false_splits,
        split_ents) = evaluate.sorted_vi_components(seg, gt)

    print "Total entropy of false merges: %f" % sum(merge_ents)
    print "Total entropy of false splits: %f" % sum(split_ents)


    worst_false_splits = worst_false_splits[:LIST_CAP]
    split_ents = split_ents[:LIST_CAP]
    worst_false_merges = worst_false_merges[:LIST_CAP]
    merge_ents = merge_ents[:LIST_CAP]

    # for label, entropies in [("merges", merge_ents), ("splits", split_ents)]:
    #     print "For worst false %s, top %d biggest entropies:" % (label, len(entropies))
    #     for rank, entropy in enumerate(entropies):
    #         print "    %d. %f" % (rank, entropy)

    # handle worst false splits
    cont_table = evaluate.contingency_table(seg, gt)
    print worst_false_splits
    # view_biggest_cross_section(gt, [worst_false_splits[0]])

    # for bad_merge_id_in_seg in worst_false_merges[1:3]:
    #     overlaps = evaluate.split_components(bad_merge_id_in_seg, cont_table, axis=0)
    #     print "bad_merge_id_in_seg:",bad_merge_id_in_seg
    #     print overlaps
    #     overlap_ids = [gt_id for gt_id, perc_of_bad_split, perc_in_bad_split in overlaps]
    #     view_bad_merge(seg, gt, bad_merge_id_in_seg, overlap_ids)

    for bad_split_id_in_gt in worst_false_splits[:1]:
        overlaps = evaluate.split_components(bad_split_id_in_gt, cont_table, axis=1)
        print "bad_split_id_in_gt:",bad_split_id_in_gt
        print overlaps
        overlap_ids = [seg_id for seg_id, perc_of_bad_split, perc_in_bad_split in overlaps]
        view_bad_split(seg, gt, bad_split_id_in_gt, overlap_ids)

    # handle worst false merges
    # for ii in range(2):


def good_color_map(idx, total):
    """ range color from green to blue """
    color_frac = float(idx)/total
    return (0,1-color_frac,color_frac)

def bad_color_map(idx, total):
    """ range color from red to blue """
    color_frac = float(idx)/total
    return (1-color_frac,0,color_frac)

def view_bad_merge(seg, gt, attempt_id, overlap_ids):
    return view_mistake(gt, seg, attempt_id, overlap_ids,
                good_color_map, bad_color_map)

def view_bad_split(seg, gt, target_id, overlap_ids):
    return view_mistake(seg, gt, target_id, overlap_ids, 
                bad_color_map, good_color_map)

def view_mistake(seg, gt, target_id, overlap_ids, seg_color_map, gt_color_map):
    fig = mayavi.mlab.gcf()

    # display attempts 
    for idx, overlap_id in enumerate(overlap_ids):
        extracted = imio.extract_segments(seg, [overlap_id])
        unsquished = np.repeat(extracted, 5, axis=0)
        color = seg_color_map(idx, len(overlap_ids))
        print "Writing contour with color "+str(color)
        mayavi.mlab.contour3d(unsquished, color=color, figure=fig)

    # display target 
    extracted = imio.extract_segments(gt, [target_id])
    unsquished = np.repeat(extracted, 5, axis=0)
    color = gt_color_map(0,1)
    mayavi.mlab.contour3d(unsquished, color=color, figure=mayavi.mlab.figure())
    
    mayavi.mlab.show()

def view_segments(seg, segment_ids):
    fig = maya.mlab.gcf()
    extracted = imio.extract_segments(seg, segment_ids)
    unsquished = np.repeat(extracted, 5, axis=0)
    mayavi.mlab.contour3d(unsquished)
    mayavi.mlab.show()

def view_biggest_cross_section(seg, segment_ids):
    extracted = imio.extract_segments(seg, segment_ids)
    biggest_index =  np.argmax(extracted.sum(axis=2).sum(axis=1))
    disp = np.zeros_like(seg)
    disp[biggest_index, :, :] = extracted[biggest_index,:,:]
    mayavi.mlab.contour3d(disp)
    mayavi.mlab.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg", type=str,
                    help="Path to the segmentation")
    parser.add_argument("gt", type=str,
                    help="Path to the groundtruth")
    parser.add_argument("--solid", action="store_true", 
                    help="show worst merge")
    parser.add_argument("--plot", action="store_true",
                    help="show vi breakdown plot")
    args = parser.parse_args()
    seg = imio.read_image_stack(args.seg)
    gt = imio.read_image_stack(args.gt)

    if args.solid:
        show_greatest_vi(seg, gt)
    if args.plot:
        plot_vi_breakdown(seg, gt)
    
if __name__ == '__main__':
    main()
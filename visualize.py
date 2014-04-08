import mayavi.mlab
import gala.viz as viz
import gala.imio as imio
import gala.evaluate as evaluate
import argparse
import numpy as np
import gala.features as features
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
    print worst_false_splits
    cont_table = evaluate.contingency_table(seg, gt)
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

def points_from_binary_volume(binary_volume, z_compression_factor=1):
    num_points = np.sum(binary_volume)
    points = np.zeros([num_points, 3], dtype=np.integer)
    print "binary_volume shape:",binary_volume.shape
    point_id = 0
    for zz in range(binary_volume.shape[0]):
        for yy in range(binary_volume.shape[1]):
            for xx in range(binary_volume.shape[2]):
                if not binary_volume[zz, yy, xx]: continue
                points[point_id, 0] = zz
                points[point_id, 1] = yy
                points[point_id, 2] = xx
                point_id += 1
    return points

def adjacent_point_values(point, volume):
    adjacent_points = np.zeros((6, 3))
    adjacent_point_values = np.zeros(6)
    for ii in range(6): adjacent_points[ii, :] = point
    current = 0
    for delta in [-1, 1]:
        for dimension in [0,1,2]:
            adjacent_points[current, dimension] += delta
    for ii in range(6):
        adjacent_point_values[ii] = volume[tuple(adjacent_points[ii, :])]
    return adjacent_point_values

def random_edge_point(points, labeled_volume):
    attempts = 0; increments = 0; MAX_ATTEMPTS = 20
    on_edge = 0
    incremental_vector = np.array([1,1,1])
    candidate = points[0,:]
    while attempts < MAX_ATTEMPTS:
        candidate = points[np.random.randint(points.shape[0]), :]
        original_seg_id = labeled_volume[tuple(candidate)]
        seg_id = original_seg_id
        while seg_id == original_seg_id and is_in_bounds(candidate, labeled_volume.shape):
            seg_id = labeled_volume[tuple(candidate)]
            candidate += incremental_vector
        if seg_id != original_seg_id: return candidate
        attempts += 1
    return points[np.random.randint(points.shape[0]), :]

def is_in_bounds(point, shape):
    for dim in range(point.shape[0]):
        if point[dim] < 0 or point[dim] >= shape[dim]: return False
    return True

def generate_points_on_vector(vector, starting_point, length, density=1):
    num_points = length * density
    points = np.zeros([num_points, 3])
    print "vector:",vector
    incremental_vector = vector.astype(np.double) / density
    print incremental_vector
    for ii in range(num_points):
        points[ii] = starting_point + (incremental_vector * ii)
    return points

def stretch_dimension(points, dimension, factor):
    points[:, dimension] *= factor

def visualize_direction(seg, segment_id):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    SUBSAMPLE_TARGET = 500
    MIN_SAMPLE_STRIDE = 10
    BOX_RADIUS = 110

    binary_volume1 = imio.extract_segments(seg, [segment_id])
    all_points_1 = points_from_binary_volume(binary_volume1)
    print all_points_1.shape
    center = random_edge_point(all_points_1, seg)
    binary_volume2 = imio.extract_segments(seg, [seg[tuple(center)]])
    all_points_2 = points_from_binary_volume(binary_volume2)
    stretch_dimension(all_points_1, 0, 5)
    stretch_dimension(all_points_2, 0, 5)

    vectors = features.direction.compute_pc_vectors(all_points_1, all_points_2, center,
                BOX_RADIUS, SUBSAMPLE_TARGET, MIN_SAMPLE_STRIDE)
    cropped_points_1 = features.direction.limit_to_radius(all_points_1, center, 
                            BOX_RADIUS, SUBSAMPLE_TARGET, MIN_SAMPLE_STRIDE)
    cropped_points_2 = features.direction.limit_to_radius(all_points_2, center, 
                            BOX_RADIUS, SUBSAMPLE_TARGET, MIN_SAMPLE_STRIDE)
    print "pc vectors:",vectors
    print "feature vector:", features.direction.compute_feature_vector(vectors)
    center_of_1 = cropped_points_1.mean(axis=0)
    center_of_2 = cropped_points_2.mean(axis=0)
    svd_1_points = generate_points_on_vector(vectors[0,:], center_of_1, BOX_RADIUS)
    svd_2_points = generate_points_on_vector(vectors[1,:], center_of_2, BOX_RADIUS)
    between_center_points = generate_points_on_vector(vectors[2,:], center_of_2, 
                np.sqrt((center_of_1-center_of_2).dot(center_of_1-center_of_2)).astype(np.integer))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for points, color, marker in [(cropped_points_2, "b", "o"), (cropped_points_1, "y", "s"),
                (svd_1_points, "g", "^"), (svd_2_points, "r", "v"), (between_center_points, "k", "<")]:
        print points.shape
        ax.scatter(points[:,2], points[:,1], points[:,0], c=color, marker=marker)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seg", type=str,
                    help="Path to the segmentation")
    parser.add_argument("gt", type=str,
                    help="Path to the groundtruth")
    parser.add_argument("--solid", action="store_true", 
                    help="show worst merge")
    parser.add_argument("--direction", action="store_true",
                    help="show directions")
    parser.add_argument("--plot", action="store_true",
                    help="show vi breakdown plot")
    args = parser.parse_args()
    seg = imio.read_image_stack(args.seg)
    gt = imio.read_image_stack(args.gt)

    if args.solid:
        show_greatest_vi(seg, gt)
    if args.direction:
        visualize_direction(gt,59)
    if args.plot:
        plot_vi_breakdown(seg, gt)
    
if __name__ == '__main__':
    main()
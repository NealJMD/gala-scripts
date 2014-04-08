import mayavi.mlab
# import gala.viz as viz
# import gala.imio as imio
# import gala.evaluate as evaluate
import argparse
import numpy as np
# import gala.features as features
# from scipy.misc import imshow
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_json(filename):
    f = open(filename, 'r')
    a = json.load(f)
    f.close()
    return a

def vectorize(p1,p2):
    return [[p1[2], p2[2]], [p1[1], p2[1]], [p1[0], p2[0]]]

def sq_distance(p1, p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2])

def show_merge_sets(merge_sets_filename):
    merge_sets = load_json(merge_sets_filename)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    skip_count = 200
    for mm in range(len(merge_sets)/2,len(merge_sets), skip_count):
        m = merge_sets[mm]
        for id1, id2, color in [('p1','p2','b'), ('p1','s1','g'), ('p2','s2','g')]:
            ax.plot(*vectorize(m[id1], m[id2]), color=color)
        # for i,j,k,h in zip(dates,zaxisvalues0,lows,highs):
        #     ax.plot([i,i],[j,j],[k,h],color = 'g')
        # ax.scatter(points[:,2], points[:,1], points[:,0], c=color, marker=marker)
    plt.show()

def shortest_path_order(ps):
    temp = np.zeros(ps.shape[1])
    champ_index = 0
    for ii in range(ps.shape[0]-1):
        champ_distance = np.inf
        for jj in range(ii, ps.shape[0]):
            dist = sq_distance(ps[ii,:], ps[jj,:])
            if sq_distance(ps[ii,:], ps[jj,:]) < champ_distance:
                champ_index = jj
                champ_distance = dist
        temp = ps[champ_index, :]
        ps[champ_index, :] = ps[ii+1, :]
        ps[ii+1,:] = temp
    return ps


def add_skeleton(point_list, ax, color=None):
    ps = np.vstack(point_list)
    print ps.shape,
    if color == None: color = np.random.rand(3,)
    # ps = shortest_path_order(ps)
    for ii in range(ps.shape[0]-1):
        ax.scatter(ps[:,2], ps[:,1], ps[:,0], c=color)
        # ax.plot(*vectorize(ps[ii], ps[ii+1]), color=color)

def show_skeletons(skeletons_filename):
    skeleton_merges = load_json(skeletons_filename)
    skeleton_merges.reverse() # reverse so that latest merges are first
    seen_segment_ids = {}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_size = 80
    total_count = 3
    count = 0
    for m in skeleton_merges:
        if m['merge_id1'] in seen_segment_ids or m['merge_id2'] in seen_segment_ids:
            continue
        if count >= total_count: break
        if len(m['centroids']) < min_size: continue
        seen_segment_ids[m['merge_id1']] = 1
        seen_segment_ids[m['merge_id2']] = 1
        add_skeleton(m['centroids'], ax)
        count += 1
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-merge-sets", type=str, default="",
                    help="show sets of four merges (pass path)")
    parser.add_argument("--show-skeletons", type=str, default="",
                    help="show sets of skeletons (pass path)")
    args = parser.parse_args()

    if len(args.show_merge_sets) > 0:
        show_merge_sets(args.show_merge_sets)
    if len(args.show_skeletons) > 0:
        show_skeletons(args.show_skeletons)

if __name__ == '__main__':
    main()
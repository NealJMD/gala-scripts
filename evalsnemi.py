import numpy as np
import h5py
import sys
from scipy.sparse import *
from scipy import *


def main():

    if len(sys.argv) < 3:
        print "Usage: python evalsnemi.py <segmentation path> <groundtruth path>"
        return

    gt = h5py.File(sys.argv[2], mode='r')
    prop = h5py.File(sys.argv[1], mode='r')

    data_gt = np.array(gt['stack'])
    data_prop = np.array(prop['stack'])
   
    prec, rec, re = rand_error(data_prop, data_gt, all_stats=True)
 
    print "Precision:", prec
    print "Recall:",rec
    print "Rand Error:",re


def rand_error(seg, gt, all_stats=False):
    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = np.array(b.todense()) ** 2

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    prec = sumAB / sumB
    rec = sumAB / sumA

    fScore = 2.0 * prec * rec / (prec + rec)
    re = 1.0 - fScore
    
    if all_stats: return (prec, rec, re)
    else: return re

if __name__ == "__main__":
    main()

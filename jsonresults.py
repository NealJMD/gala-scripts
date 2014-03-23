import gala.imio as io
from rungala import get_paths
import sys
import json
from evalsnemi import rand_error

SIZE = "quarter"
ALL_VOLUME_IDS = ["topleft", "topright", "bottomleft", "bottomright"]
ALL_FEATURES = ["base", "baseANDinclusion", "baseANDcontact", "baseANDinclusionANDcontact", "baseANDsquiggliness", "baseANDinclusionANDcontactANDsquiggliness"]
ALL_CUES = ["idsia", "idsiaANDder"]

def create_results_dict():
    results = {}
    for test_set in ALL_VOLUME_IDS:
        if test_set not in results: results[test_set] = {}
        gt_path = get_paths("gala-evaluate", "train", SIZE, 
                    test_set, "XX", "XX", "XX", "XX")["groundtruth"]
        gt = io.read_image_stack(gt_path)
        for train_set in ALL_VOLUME_IDS:
            if train_set not in results[test_set]: results[test_set][train_set] = {}
            for feature in ALL_FEATURES:
                if feature not in results[test_set][train_set]:
                    results[test_set][train_set][feature] = {}
                    for cue in ALL_CUES:
                        seg_path = get_paths("gala-evaluate", "train", SIZE,
                            test_set, cue, feature, "", train_set)["segmentation"]
                        try: 
                            seg = io.read_image_stack(seg_path)
                            err = rand_error(seg,gt)
                            print "test %s, train %s, %s, %s: %f" % (test_set, 
                                            train_set, feature, cue, err)
                            results[test_set][train_set][feature][cue] = err
                        except:
                            print " -- missing: test %s, train %s, %s, %s: %s" % (
                            test_set, train_set, feature, cue, seg_path)
    return results
    
def main():
    results = create_results_dict()
    print json.dumps(results, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()

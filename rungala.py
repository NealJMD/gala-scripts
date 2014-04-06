""" Standardize the inputs, outputs, and logs of running GALA """

import runner
import os, sys
import subprocess as sp
import datetime
from evalsnemi import rand_error
from gala.imio import read_image_stack

opj = os.path.join # convenience alias
GLOBAL_PREFIX = "/n/fs/neal-thesis/"
DATA_PREFIX = opj(GLOBAL_PREFIX, "data")
BIN_PREFIX = opj(GLOBAL_PREFIX, "gala/bin/")
H5_EXT = ".lzf.h5"
CLASSIFIER_EXT = ".classifier.joblib"
SEGMENTATION_EXT = "-0.50.lzf.h5"
ID_DELIMITER = "AND"
FILENAME_PART_DELIMITER = "_"
ALL_VOLUME_IDS = ["topleft", "topright", "bottomleft", "bottomright"]
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DEFAULT_CLASSIFIER_VOLUME = "quarter"

def get_features_from_id(features_id):

    features =  "features.base.Composite(children=[\n"
    ending = "])"

    # watch out, I'm using features_id and feature_ids
    feature_ids = features_id.split(ID_DELIMITER)
    recognized_feature_ids = []
    if "base" in feature_ids:
        features += """features.moments.Manager(),
                   features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9]),
                   features.graph.Manager()"""
        recognized_feature_ids.append("base")
    if "inclusion" in feature_ids:
        features += ",\nfeatures.inclusion.Manager()"
        recognized_feature_ids.append("inclusion")
    if "squiggliness" in feature_ids:
        features += ",\nfeatures.squiggliness.Manager()"
        recognized_feature_ids.append("squiggliness")
    if "contact" in feature_ids:
        features += ",\nfeatures.contact.Manager([0.1, 0.5, 0.9])"
        recognized_feature_ids.append("contact")
    if "direction" in feature_ids:
        features += ",\nfeatures.direction.Manager(5, 110, 500, 10)"
        recognized_feature_ids.append("direction")
    if "stage" in feature_ids:
        features += ",\nfeatures.stage.Manager()"
        recognized_feature_ids.append("stage")
    if "skeleton3n" in feature_ids:
        features += ",\nfeatures.skeleton.Manager(5,3)"
        recognized_feature_ids.append("skeleton3n")
    if "skeleton2n" in feature_ids:
        features += ",\nfeatures.skeleton.Manager(5,2)"
        recognized_feature_ids.append("skeleton2n")
    if "skeleton" in feature_ids:
        features += ",\nfeatures.skeleton.Manager(5, 1)"
        recognized_feature_ids.append("skeleton")
    if set(feature_ids) != set(recognized_feature_ids):
        raise KeyError("Unrecognized feature_id! Of %s, recognized %s" % (
                str(feature_ids), str(recognized_feature_ids)))

    features += ending
    return features


def filename_join(*parts):
    return FILENAME_PART_DELIMITER.join(parts)


def ensure_path(path):
    if os.path.exists(path): return path
    os.makedirs(path)
    return path


def get_specified_file_path(specifier, ext):
    filename = filename_join(*specifier) + ext
    subfolder = opj(*specifier)
    return opj(DATA_PREFIX, subfolder, filename)


def get_logfiles(log_dir, experiment_name, err_to_out=True):
    logfiles = {}
    base_filename = filename_join(experiment_name, TIMESTAMP)
    ensure_path(log_dir)
    logfiles["stdout"] = opj(log_dir, filename_join(base_filename, "stdout"))
    logfiles["performance"] = opj(log_dir, filename_join(base_filename, "performance"))
    if err_to_out:
        logfiles["stderr"] = sp.STDOUT
    else:
        logfiles["stderr"] = opj(log_dir, filename_join(base_filename, "stderr"))
    return logfiles


def print_paths(paths):
    for key, path in paths.iteritems():
        print " -  (%s): %s" % (key, path)
    return
 

def get_paths(task, traintest, size, volume_id, cues_id, features_id, exec_id="", classifier_id=""):
    paths = {}
    classifier_name = "classifer-"+classifier_id

    # command
    paths["command"] = opj(BIN_PREFIX, task)

    # watersheds
    specifier = ["watersheds", traintest, size, volume_id]
    paths["watersheds"] = get_specified_file_path(specifier, H5_EXT)

    # groundtruth
    specifier = ["groundtruth", traintest, size, volume_id]
    paths["groundtruth"] = get_specified_file_path(specifier, H5_EXT)

    # hypercubes - 4 dimensional images - height, width, frame count, channels
    specifier = ["hypercubes", traintest, size, volume_id, cues_id]
    paths["hypercubes"] = get_specified_file_path(specifier, H5_EXT)

    # output
    specifier = ["output", traintest, size, volume_id, cues_id, features_id, task]
    if len(classifier_id): specifier.append(classifier_name)
    if len(exec_id) > 0: specifier.append(exec_id)
    paths["output_dir"] = opj(DATA_PREFIX, *specifier)
    paths["experiment_name"] = filename_join(*specifier)
    paths["log_dir"] = opj(paths["output_dir"], "logs")

    # classifier - for gala-segment
    specifier = ["output", "train", DEFAULT_CLASSIFIER_VOLUME, classifier_id, cues_id, features_id, "gala-train"]
    if len(exec_id) > 0: specifier.append(exec_id)
    paths["classifier"] = get_specified_file_path(specifier, CLASSIFIER_EXT)

    # segmentation - for gala-evaluate
    specifier = ["output", traintest, size, volume_id, cues_id, features_id, "gala-segment", classifier_name]
    paths["segmentation"] = get_specified_file_path(specifier, SEGMENTATION_EXT)
    
    return paths


def run_gala_train(traintest, size, volume_id, cues_id, features_id, exec_id="", *args):
    """
    Parameters
    ----------
    traintest: {"train", "test"} whether to train on the training or testing
        dataset (throws error if it's "test")
    size: the size of the dataset used to train the classifier
    cues_id: a '+' separated list of cues used in training and segmentation
    features_id: the id of the features used in training and segmentation
    """
    if traintest == "test": raise RuntimeError("Do not train on test data!")

    paths = get_paths("gala-train", traintest, size, volume_id, cues_id, features_id, exec_id)
    print_paths(paths)
    command = paths["command"]
    logfiles = get_logfiles(paths["log_dir"], paths["experiment_name"], err_to_out=True)
    positionals = [paths["hypercubes"], paths["groundtruth"]]
    gala_options = {}
    gala_options["--feature-manager"] = get_features_from_id(features_id)
    gala_options["--watershed-file"] = paths["watersheds"]
    gala_options["--experiment-name"] = paths["experiment_name"]
    gala_options["--output-dir"] = ensure_path(paths["output_dir"])
    gala_options["--verbose"] = ""
    gala_options["--show-progress"] = ""

    # gala_options["--profile"] = ""
    # print "---\nProfiling with cProfile!\n---"

    channel_count = len(cues_id.split(ID_DELIMITER))
    if channel_count < 2:
        gala_options["--no-channel-data"] = ""
        gala_options["--single-channel"] = ""

    return runner.call_and_monitor_command(command, positionals, gala_options, logfiles)


def run_gala_segment(traintest, size, volume_id, cues_id, features_id, exec_id="", classifier_volume_id=""):
    """
    Parameters
    ----------
    traintest: {"train", "test"} whether to segment the training or testing
        dataset
    size: the size of the dataset being segmented and that trained the classifier
    cues_id: a '+' separated list of cues used in training and segmentation
    features_id: the id of the features used in training and segmentation
    """

    if volume_id == "all":
        if len(classifier_volume_id) < 1:
            raise ValueError("Need classifier_id to run on all volumes!")
        children = []
        for new_volume_id in ALL_VOLUME_IDS:
            child = run_gala_segment(traintest, size, new_volume_id, cues_id, 
                            features_id, exec_id, classifier_volume_id)
            children += child
        return children

    if len(classifier_volume_id) < 1:
        classifier_volume_id = volume_id
        print "tesing on training data"

    paths = get_paths("gala-segment", traintest, size, volume_id, cues_id, features_id, exec_id, classifier_volume_id)
    print_paths(paths)
    command = paths["command"]
    logfiles = get_logfiles(paths["log_dir"], paths["experiment_name"], err_to_out=True)
    positionals = [paths["hypercubes"]]

    gala_options = {}
    gala_options["--feature-manager"] = get_features_from_id(features_id)
    gala_options["--watershed"] = paths["watersheds"]
    gala_options["--experiment-name"] = paths["experiment_name"]
    gala_options["--output-dir"] = ensure_path(paths["output_dir"])
    gala_options["--verbose"] = ""
    gala_options["--show-progress"] = ""

    channel_count = len(cues_id.split(ID_DELIMITER))
    if channel_count < 2:
        gala_options["--no-channel-data"] = ""
        gala_options["--single-channel"] = ""

    gala_options["--classifier"] = paths["classifier"]
    gala_options["--thresholds"] = "0.5" #" ".join([str(x/10.) for x in range(0,11)]) # 0-1 by 0.1
    gala_options["--no-raveler-export"] = ""
    gala_options["--no-graph-json"] = ""

    child = runner.call_command(command, positionals, gala_options, logfiles)
    return [child]
    #return runner.call_and_monitor_command(command, positionals, gala_options, logfiles)


def run_gala_evaluate(traintest, size, volume_id, cues_id, features_id, exec_id, classifier_volume_id):
    """
    Parameters
    ----------
    traintest: {"train", "test"} whether to evaluate the output of the
        segmentation of the training or testing data
    size: the size of the dataset being evaluated and the groundtruth
    cues_id: a '+' separated list of cues used in training and segmentation
    features_id: the id of the features used in training and segmentation
    """
    if classifier_volume_id == "all":
        for new_cl_volume_id in ALL_VOLUME_IDS:
            run_gala_evaluate(traintest, size, volume_id, cues_id, 
                            features_id, exec_id, new_cl_volume_id)
        return

    if volume_id == "all":
        for new_volume_id in ALL_VOLUME_IDS:
            run_gala_evaluate(traintest, size, new_volume_id, cues_id, 
                            features_id, exec_id, classifier_volume_id)
        return
    
    paths = get_paths("gala-evaluate", traintest, size, volume_id, cues_id, features_id, exec_id, classifier_volume_id)

    for file_use in ["segmentation", "groundtruth"]:
        if not os.path.isfile(paths[file_use]):
            print "Unable to find %s file for %s %s %s %s %s classifier tested on %s" % (
                file_use, classifier_volume_id, size, cues_id, features_id, exec_id, volume_id)
            return

    seg = read_image_stack(paths["segmentation"])
    gt = read_image_stack(paths["groundtruth"])

    error = rand_error(seg, gt)

    print "For classifier trained on %s %s %s %s %s and tested on %s %s" % (
            classifier_volume_id, size, cues_id, features_id, exec_id, volume_id, size)
    print "\tfound Rand error: %f" % (error)

#    return child.wait()


def main(args):
    """ 
    example: python rungala.py gala-train train 128x64x32 idsia base
              python rungala.py gala-segment test 128x64x32 idsia base
              python rungala.py gala-evaluate test 128x64x32 idsia base

    example: python rungala.py gala-train train half idsia+idsia_der+luminance_der base
             python rungala.py gala-segment test half idsia+idsia_der+luminance_der base
             python rungala.py gala-evaluate test half idsia+idsia_der+luminance_der base
    """

    if len(args) < 6: 
        print "Usage: python rungala.py <gala-train|gala-segment|gala-evaluate> \
<train|test> <size> <volume_id> <cues> <features> \
[execution_id] [classifier_volume_id] [dry_run]"
        return
    task = args[1]
    gala_args = args[2:]
    if len(args) > 9: 
        paths = get_paths(task, *gala_args[:-1])
        print_paths(paths)
        return
    if task == "gala-train":
        status = run_gala_train(*gala_args)
        print "gala-train call exited with status %d" % (status)
    elif task == "gala-segment": # supports parallelism
        children = run_gala_segment(*gala_args)
        for child in children: child.wait()
        return
    elif task == "gala-evaluate":
        return run_gala_evaluate(*gala_args)
    elif task == "gala-all":
        traintest, size, volume_id, cues_id, features_id, exec_id, classifier_volume_id = tuple(gala_args)
        run_gala_train(traintest, size, volume_id, cues_id, features_id, exec_id,"")
        chilren = run_gala_segment(traintest, size, "all", cues_id, features_id, exec_id, volume_id)
        for child in children: child.wait()
        run_gala_evaluate(traintest, size, "all", cues_id, features_id, exec_id, volume_id)
    else: raise RuntimeError("Unknown task: %s" % (task))

if __name__ == '__main__':
    main(sys.argv)

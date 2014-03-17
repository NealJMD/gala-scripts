""" Standardize the inputs, outputs, and logs of running GALA """

import runner
import os, sys
import subprocess as sp
import datetime

opj = os.path.join # convenience alias
GLOBAL_PREFIX = "/n/fs/neal-thesis/"
DATA_PREFIX = opj(GLOBAL_PREFIX, "data")
BIN_PREFIX = opj(GLOBAL_PREFIX, "gala/bin/")
H5_EXT = ".lzf.h5"
CLASSIFIER_EXT = ".classifier.joblib"
ID_DELIMITER = "AND"
FILENAME_PART_DELIMITER = "_"
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_features_from_id(features_id):

    base =  """features.base.Composite(children=[
                   features.moments.Manager(),
                   features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9]),
                   features.graph.Manager()"""
    ending = "])"
    if features_id == "base":
        return base + ending
    elif features_id == "baseANDinclusion":
        return base + ",\nfeatures.inclusion.Manager()" + ending
    else:
        raise KeyError("Unrecognized feature_id: %s" % (features_id))


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

    # command
    paths["command"] = opj(BIN_PREFIX, task)

    # watersheds
    specifier = ["watersheds", traintest, size, volume_id]
    paths["watersheds"] = get_specified_file_path(specifier, H5_EXT)

    # groundtruth
    if task == "gala-evaluate": specifier = ["groundtruth", "test", size]
    else: specifier = ["groundtruth", traintest, size, volume_id]
    paths["groundtruth"] = get_specified_file_path(specifier, H5_EXT)

    # hypercubes - 4 dimensional images - height, width, frame count, channels
    specifier = ["hypercubes", traintest, size, volume_id, cues_id]
    paths["hypercubes"] = get_specified_file_path(specifier, H5_EXT)

    # output
    specifier = ["output", traintest, size, volume_id, cues_id, features_id, task]
    if len(classifier_id): specifier.append("classifer-"+classifier_id)
    if len(exec_id) > 0: specifier.append(exec_id)
    paths["output_dir"] = opj(DATA_PREFIX, *specifier)
    paths["experiment_name"] = filename_join(*specifier)
    paths["log_dir"] = opj(paths["output_dir"], "logs")

    # classifier - for gala-segment
    specifier = ["output", "train", size, classifier_id, cues_id, features_id, "gala-train"]
    paths["classifier"] = get_specified_file_path(specifier, CLASSIFIER_EXT)

    # segmentation - for gala-evaluate
    specifier = ["output", traintest, size, volume_id, cues_id, features_id, "gala-segment"]
    paths["segmentation"] = "UNIMPLEMENTED" # need to see how it is spit out

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
    gala_options["--thresholds"] = " ".join([str(x/10.) for x in range(0,11)]) # 0-1 by 0.1
    gala_options["--no-raveler-export"] = ""
    gala_options["--no-graph-json"] = ""
    return runner.call_and_monitor_command(command, positionals, gala_options, logfiles) 


def run_gala_evaluate(traintest, size, volume_id, cues_id, features_id, exec_id=""):
    """
    Parameters
    ----------
    traintest: {"train", "test"} whether to evaluate the output of the
        segmentation of the training or testing data
    size: the size of the dataset being evaluated and the groundtruth
    cues_id: a '+' separated list of cues used in training and segmentation
    features_id: the id of the features used in training and segmentation
    """
    paths = get_paths("gala-evaluate", traintest, size, volume_id, cues_id, features_id, exec_id)
    print_paths(paths)
    command = paths["command"]
    logfiles = get_logfiles(paths["log_dir"], paths["experiment_name"], err_to_out=True)
    positionals = [paths["segmentation"], paths["groundtruth"]]

    gala_options["--rand-index"] = ""
    gala_options["--threshold"] = 0.5

    child = runner.call_command(command, positionals, gala_options, logfiles)
    return child.wait()


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
    if task == "gala-train": run_gala = run_gala_train
    elif task == "gala-segment": run_gala = run_gala_segment
    elif task == "gala-evaluate": run_gala = run_gala_evaluate
    else: raise RuntimeError("Unknown task: %s" % (task))

    return run_gala(*gala_args)

if __name__ == '__main__':
    main(sys.argv)

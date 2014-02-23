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
ID_DELIMITER = "+"
FILENAME_PART_DELIMITER = "_"
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_features_from_id(features_id):
    if features_id == "base":
        return """features.base.Composite(children=[
                   features.moments.Manager(),
                   features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9]),
                   features.graph.Manager()])"""
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


def print_paths(*path_args):
    paths = get_paths(*path_args)
    for key, path in paths.iteritems():
        print " -  (%s): %s" % (key, path)
    return
 

def get_paths(task, traintest, size, cues_id, features_id):
    paths = {}

    # command
    paths["command"] = opj(BIN_PREFIX, task)

    # watersheds
    specifier = ["watersheds", traintest, size]
    paths["watersheds"] = get_specified_file_path(specifier, H5_EXT)

    # groundtruth
    if task == "gala-evaluate": specifier = ["groundtruth", "test", size]
    else: specifier = ["groundtruth", traintest, size]
    paths["groundtruth"] = get_specified_file_path(specifier, H5_EXT)

    # hypercubes - 4 dimensional images - height, width, frame count, channels
    specifier = ["hypercubes", traintest, size, cues_id]
    paths["hypercubes"] = get_specified_file_path(specifier, H5_EXT)

    # output
    specifier = ["output", traintest, size, cues_id, features_id, task]
    paths["output_dir"] = opj(DATA_PREFIX, *specifier)
    paths["experiment_name"] = filename_join(*specifier)
    paths["log_dir"] = opj(paths["output_dir"], "logs")

    # classifier - for gala-segment
    specifier = ["output", "train", size, cues_id, features_id, "gala-train"]
    paths["classifier"] = get_specified_file_path(specifier, CLASSIFIER_EXT)

    # segmentation - for gala-evaluate
    specifier = ["output", traintest, size, cues_id, features_id, "gala-segment"]
    paths["segmentation"] = "UNIMPLEMENTED" # need to see how it is spit out

    return paths


def run_gala_train(traintest, size, cues_id, features_id):
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

    paths = get_paths("gala-train", traintest, size, cues_id, features_id)
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

    channel_count = len(cues_id.split(ID_DELIMITER))
    if channel_count < 2:
        gala_options["--no-channel-data"] = ""
        gala_options["--single-channel"] = ""

    return runner.call_and_monitor_command(command, positionals, gala_options, logfiles)


def run_gala_segment(traintest, size, cues_id, features_id):
    """
    Parameters
    ----------
    traintest: {"train", "test"} whether to segment the training or testing
        dataset
    size: the size of the dataset being segmented and that trained the classifier
    cues_id: a '+' separated list of cues used in training and segmentation
    features_id: the id of the features used in training and segmentation
    """

    paths = get_paths("gala-segment", traintest, size, cues_id, features_id)
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


def run_gala_evaluate(traintest, size, cues_id, features_id):
    """
    Parameters
    ----------
    traintest: {"train", "test"} whether to evaluate the output of the
        segmentation of the training or testing data
    size: the size of the dataset being evaluated and the groundtruth
    cues_id: a '+' separated list of cues used in training and segmentation
    features_id: the id of the features used in training and segmentation
    """
    paths = get_paths("gala-evaluate", traintest, size, cues_id, features_id)
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
        print "Usage: python rungala.py <gala-train|gala-segment|gala-evaluate> <train|test> <size> <cues> <features> [dry_run]"
        return
    script, task, traintest, size, cues_id, features_id = tuple(args[0:6])
    if len(args) > 6: 
        print_paths(task, traintest, size, cues_id, features_id)
        return
    if task == "gala-train": run_gala = run_gala_train
    elif task == "gala-segment": run_gala = run_gala_segment
    elif task == "gala-evaluate": run_gala = run_gala_evaluate
    else: raise RuntimeError("Unknown task: %s" % (task))

    return run_gala(traintest, size, cues_id, features_id)

if __name__ == '__main__':
    main(sys.argv)

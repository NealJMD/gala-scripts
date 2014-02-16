""" Standardize the inputs, outputs, and logs of running GALA """


import runner
import os, sys
import subprocess as sp
import datetime

opj = os.path.join # convenience alias
GLOBAL_PREFIX = "/n/fs/neal-thesis/"
BIN_PREFIX = opj(GLOBAL_PREFIX, "gala/bin/")
INPUT_PREFIX = opj(GLOBAL_PREFIX, "data/input/")
OUTPUT_PREFIX = opj(GLOBAL_PREFIX, "data/output")
LOG_PREFIX = opj(GLOBAL_PREFIX, "data/logs")
BOUNDARY_SUBDIR = "membrane_probabilities"
LABEL_SUBDIR = "labels"
WATERSHED_SUBDIR = "watersheds2d"
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d__%H:%M:%S")

def get_base_features():
    return "features.base.Composite(children=[\ \
                       features.moments.Manager(), \ \
                       features.histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9]), \ \
                       features.graph.Manager()])"

def generate_name(set_size, input, features):
    return "__".join([set_size, input, features, TIMESTAMP])


def get_logfiles(name, subfolder="", err_to_out=True):
    logfiles = {}
    logfiles["stdout"] = ensure_file(opj(LOG_PREFIX, subfolder, name))
    if err_to_out: logfiles["stderr"] = sp.STDOUT
    else: logfiles["stderr"] = ensure_file(logfiles["stdout"] + "_err")
    logfiles["performance"] = ensure_file(opj(LOG_PREFIX, "performance", subfolder, name))
    return logfiles

def ensure_path(path):
    if os.path.exists(path): return path
    os.makedirs(path)
    return path

def ensure_file(file_path):
    ensure_path(os.path.dirname(file_path))
    return file_path

def run_gala_train(boundary_file, label_file, watershed_file, experiment_name,
                   gala_options={}, logfiles={}):
    command = opj(BIN_PREFIX, "gala-train")
    boundary_maps = opj(INPUT_PREFIX, BOUNDARY_SUBDIR, boundary_file)
    training_labels = opj(INPUT_PREFIX, LABEL_SUBDIR, label_file)
    positionals = [boundary_maps, training_labels]
    output_dir = ensure_path(opj(OUTPUT_PREFIX, experiment_name))
    gala_options["--verbose"] = ""
    gala_options["--show-progress"] = ""
    gala_options["--output-dir"] = output_dir 
    gala_options["--watershed-file"] = opj(INPUT_PREFIX, WATERSHED_SUBDIR, watershed_file)
    gala_options["--experiment-name"] = experiment_name

    return runner.call_and_monitor_command(command, positionals, gala_options, logfiles)
    

def run_gala_train_on_tiny():
    size = "runner-test-128x64x32"
    name = generate_name(size, "membrane-probs", "base")
    boundary_file = size+"/train-membranes-"+size+".lzf.h5"
    label_file = size+"/train-labels-"+size+".lzf.h5"
    watershed_file = size+"/watershed-2d-"+size+".lzf.h5"

    options = {}
    options["--feature-manager"] = get_base_features()
    options["--no-channel-data"] = ""
    options["--single-channel"] = ""
    logfiles = get_logfiles(name, size, True)
    run_gala_train(boundary_file, label_file, watershed_file, name, options, logfiles)

def main(args):
    if len(args) < 2: print "Usage: python rungala.py <runtype>"
    elif args[1] == "tiny": run_gala_train_on_tiny()
    else: print "Unknown runtype %s" % (args[1])

if __name__ == "__main__":
    main(sys.argv)


import psutil
import subprocess as sp
import time
import datetime


def call_and_monitor_command(command, positionals=[], options={}, logfiles={},
        Popen_options={}, poll_delay=1):
    """
    Run a command as specified. Log information about memory usage.
    This command is blocking and returns the return code of the command.

    Parameters: same as call_command.
    """
    if "performance" not in logfiles:                                        
        raise KeyError("call_and_monitor_usage needs 'performance' logfile")
    f = open(logfiles["performance"], 'w')
    f.close() # test to make sure logfile works
    child = call_command(command, positionals=positionals, options=options,
                             logfiles=logfiles, Popen_options=Popen_options)
    memory_usage = []
    while child.poll() == None:
        ram_useage, current_memory_usage = child.get_memory_info()
        memory_usage.append(str(current_memory_usage))
        time.sleep(poll_delay)
    finish_time = time.time()
    running_time = child.create_time - finish_time
    start_time = datetime.datetime.fromtimestamp(child.create_time).strftime("%Y-%m-%d %H:%M:%S")
    output = "Start time: %s\nCommand: %s\nPositionals: %s\nOption: s%s\n \
Logfiles: %s\nPopen_options: %s\nRunning time (seconds):%f\n \
Polling memory every %i seconds\n%s" %  (start_time, command, 
            str(positionals), str(options),str(logfiles),
            str(Popen_options), running_time, poll_delay,
            "\n".join(memory_usage))
    f = open(logfiles["performance"], 'w')
    f.write(output)
    f.close()
    return child.returncode

def call_command(command, positionals=[], options={}, logfiles={}, Popen_options={}):
    """
    Run a command as specified in a non-blocking way.

    Parameters
    ----------
    command: (string) command to be run
    positionals: (list of strings) positional arguments
    options: (dict of strings) all flags to be passed. val follows key
    logfiles: (dict of strings) specifies where to log. for logging, must have
        keys 'stdout' and 'stderr' pointing to files. needs 'performance' if 
        called with call_and_monitor_command.
    Popen_options: (dict of strings) options passed as kwargs to psutil.Popen

    Returns
    ------
    a psutil child process object
    """
    formatted_command = [command] + positionals
    for option, value in options.iteritems():
        formatted_command.append(option)
        if len(value) < 1: continue
        formatted_command.append(value)
    for stream_name, filename in logfiles.iteritems():
        if stream_name not in ["stdout", "stderr"]: continue
        if filename == sp.STDOUT:
            Popen_options[stream_name] = filename
            continue
        f = open(filename, 'a')
        Popen_options[stream_name] = f
        Popen_options["close_fds"] = True
    print "Calling: %s" % (" ".join(formatted_command))
    if "stdout" in logfiles:
        print "Logging stdout to %s" % (logfiles["stdout"])
    child = psutil.Popen(formatted_command, **Popen_options)
    return child

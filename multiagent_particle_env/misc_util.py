# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
misc_util.py
"""

import collections.abc as collections
import gym
import json
import numpy as np
import os
import pickle
import random
import tempfile
import time
import yaml
import zipfile

from collections import namedtuple
from colorama import Fore, Style

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2022, Multi-Agent Particle Environment'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '1.0.0'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rfernandez@utexas.edu'
__status__ = 'Dev'


def boolean_flag(parser, name, default=False, help_msg=None):
    """
    Add a boolean flag to argparse parser.

    Args:
        parser (argparse.Parser):  Parser to add the flag to
        name (str):                --<name> will enable the flag, while --no-<name> will disable it
        default (bool):            Default value of the flag (default: False)
        help_msg (str):            Help string for the flag (default: None)
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help_msg)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def dict2namedtuple(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def get_last_checkpoint_file(dir_name):
    """
    Retrieve the last checkpoint file path in the checkpoints directory

    Args:
        dir_name (str): Checkpoints directory
    Returns:
        last_checkpoint_path (str): Path to last checkpoint
        timestep (str):             Timestep corresponding to last checkpoint
        episode (str):              Episode corresponding to last checkpoint
    """
    list_of_files = os.listdir(dir_name)
    files = []
    for entry in list_of_files:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            tmp_files = get_list_of_checkpoints(full_path)
            files = files + tmp_files
        else:
            files.append(full_path)

    last_checkpoint_path = sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split('_')[2]))[-1]
    split_basename = os.path.splitext(os.path.basename(last_checkpoint_path))[0].split('_')
    return last_checkpoint_path, int(split_basename[2]), int(split_basename[4])


def get_list_of_checkpoints(dir_name):
    """
    Retrieve the list checkpoint files in the checkpoints directory

    Args:
        dir_name (str): Checkpoints directory
    Returns:
        files (list): Sorted list of checkpoints
    """
    list_of_files = os.listdir(dir_name)
    files = []
    for entry in list_of_files:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            files = files + get_list_of_checkpoints(full_path)
        else:
            files.append(full_path)

    return sorted(files, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split('_')[2]))


def get_list_of_dirs(dir_name):
    list_of_dirs = os.listdir(dir_name)
    dirs = []
    for entry in list_of_dirs:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            dirs.append(full_path)
        else:
            pass
    return sorted(dirs)


def get_wrapper_by_name(env, classname):
    """
    Given a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    Args:
        env (gym.Env):   Gym environment of gym.Wrapper
        classname (str): Name of the wrapper

    Returns:
        wrapper (gym.Wrapper): Wrapper named classname
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("[qcgraph.common.misc_util.get_wrapper_by_name()]: "
                             f"Couldn't find wrapper named {classname}")


def load_yaml_config(config_file_path):
    """
    Load and parse yaml config file

    Args:
        config_file_path (str): Path to the yaml config file

    Returns:
        config_dict (dict): Dictionary with configuration parameters
    """
    # Create loader
    loader = yaml.SafeLoader

    # Custom tag handler for joining strings
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    # Register the join tag handler
    loader.add_constructor('!join', join)

    # Load and parse config file
    with open(config_file_path, "r") as config_file:
        try:
            config_dict = yaml.load(config_file, Loader=loader)
        except yaml.YAMLError:
            raise yaml.YAMLError("[qcgraph.common.misc_utils.load_yaml_config()]: "
                                 f"Error with config file, {config_file_path}")
    return config_dict


def pickle_load(path, compression=False):
    """
    Unpickle a possible compressed pickle.

    Args:
        path (str):         Path to the output file
        compression (bool): If true, assumes that pickle was compressed when created and attempts decompression.
                            (default: False)

    Returns:
        obj (object): The unpickled object
    """

    if compression:
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:
            with myzip.open("data") as f:
                return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def pretty_eta(seconds_left):
    """
    Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Args:
        seconds_left (int): Number of seconds to be converted to the ETA
    Returns:
        eta (str): String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    # Days left
    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg

    # Hours left
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg

    # Minutes left
    if minutes_left > 0:
        return helper(minutes_left, 'minute')

    return 'less than a minute'


def recursive_dict_update(dictionary, updates):
    """
    Recursively updates a dictionary with another update dictionary

    Args:
        dictionary (dict/collections.Mapping): Dictionary to be updated
        updates (dict/collections.Mapping):    Dictionary containing the updates

    Returns:
        (dict) Updated dictionary

    """
    dictionary = dictionary.copy()
    for key, value in updates.items():
        if isinstance(value, collections.Mapping):
            # Value is a dictionary
            dictionary[key] = recursive_dict_update(dictionary.get(key, {}), value)
        else:
            dictionary[key] = value

    return dictionary


def relatively_safe_pickle_dump(obj, path, compression=False):
    """
    This is just like regular pickle dump, except from the fact that failure cases are
    different:

        - It's never possible that we end up with a pickle in corrupted state.
        - If a there was a different file at the path, that file will remain unchanged in the
          even of failure (provided that filesystem rename is atomic).
        - it is sometimes possible that we end up with useless temp file which needs to be
          deleted manually (it will be removed automatically on the next function call)

    The intended use case is periodic checkpoints of experiment state, such that we never
    corrupt previous checkpoints if the current one fails.

    Args:
        obj (object): Object to pickle
        path (str): Path to the output file
        compression (bool): If true, pickle will be compressed
    """
    temp_storage = path + ".relatively_safe"
    if compression:
        # Using gzip here would be simpler, but the size is limited to 2GB
        with tempfile.NamedTemporaryFile() as uncompressed_file:
            pickle.dump(obj, uncompressed_file)
            uncompressed_file.file.flush()
            with zipfile.ZipFile(temp_storage, "w", compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(uncompressed_file.name, "data")
    else:
        with open(temp_storage, "wb") as f:
            pickle.dump(obj, f)

    os.rename(temp_storage, path)


def save_config(config, save_path):
    """
    Save config dictionary to file in yaml and json format

    Args:
        config (dict):   Dictionary containing the configuration parameters
        save_path (str): Path to save the config file

    """
    with open(os.path.join(save_path, "config.yaml"), "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    with open(os.path.join(save_path, "config.json"), "w") as json_file:
        json.dump(config, json_file, sort_keys=True, indent=4)


def set_global_seeds(seed):
    """
    Set the seed for python random, tensorflow, torch, numpy and gym spaces

    Args:
        seed: (int) The seed
    """
    #PyTorch
    try:
        import torch
        torch.random.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        print(f'{Fore.YELLOW} [qcgraph.common.misc_util.(set_global_seeds()]: '
              f'Torch package not found, random seed not set. {Style.RESET_ALL}')
        pass

    # Numpy
    np.random.seed(seed)

    # Python
    random.seed(seed)


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value {!r}".format(val))


def time_str(in_seconds):
    """
    Convert seconds to a human-readable string showing days, hours, minutes and seconds

    Args:
        in_seconds (float/int): Seconds to be converted to human-readable format

    Returns:
        string (str): Human-readable time string for given seconds
    """
    days, remainder = divmod(in_seconds, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""

    # Days
    if days > 0:
        string += f"{int(days):d} days, "

    # Hours
    if hours > 0:
        string += f"{int(hours):d} hours, "

    # Minutes
    if minutes > 0:
        string += f"{int(minutes):d} minutes, "

    # Seconds
    string += f"{int(seconds):d} seconds"

    return string


def time_left(start_time, t_start, t_current, t_max):
    """
    Calculate time left of a process

    Args:
        start_time (float/int): Start time in seconds since the epoch
        t_start (int):          Starting timestep or other time representation
        t_current (int):        Current timestep or other time representation
                                (Must be same time representation as t_start)
        t_max (int):            Max timestep or other time representation
                                (Must be same time representation as t_start)

    Returns:
        Human-readable string representing the time left of a process
    """
    if t_current >= t_max:
        return "-"

    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)

    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)

    return time_str(time_left)


def zipsame(*seqs):
    """
    Performes a zip function, but asserts that all zipped elements are of the same size

    Args:
        seqs (list): A list of arrays that are zipped together

    Returns:
        The zipped arrays
    """
    length = len(seqs[0])
    assert all(len(seq) == length for seq in seqs[1:])
    return zip(*seqs)

from typing import NamedTuple
import os

here = os.path.dirname(__file__)
sum_rule_config_path = os.path.join(here, "runinfo_parser", "summarize_config.json")
notebook_folder = os.path.join(here, 'target_nbs')

max_len_value = 5


class var_type(NamedTuple):
    error = 'error',
    unknown = 'unknown',
    var = 'variable',
    con = 'constant',
    call = 'call',
    tuple = 'tuple',
    list = 'list',
    set = 'set',
    dict = 'dict',
    nparray = 'np.array',
    tensor = 'tf.tensor',


class var_val(NamedTuple):
    error = 'error',
    unknown = 'unknown',


class var_shape(NamedTuple):
    error = 'error',
    unknown = 'unknown',


class rule_res(NamedTuple):
    unsupported = -1,
    success = 0,
    log = 1,
    warning = 2,
    error = 3,


class notify_level(NamedTuple):
    ignore = -1,
    log = 0,
    warning = 1,
    error = 2

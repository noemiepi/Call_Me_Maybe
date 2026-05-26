from enum import Enum


class State(Enum):
    FUNCTION = "function"
    PARAMETER = "parameter"
    STRPARAM = "str_param"
    INTPARAM = "int_param"
    DECODE = "decode"
    FINAL = "final"

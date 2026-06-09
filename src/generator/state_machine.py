from enum import Enum


class State(Enum):
    FUNCTION = "function"
    PARAMETER = "parameter"
    INCOMPLETE_PARAMETER = "incomplete parameter"
    DECODE = "decode"
    FINAL = "final"

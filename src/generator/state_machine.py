from enum import Enum


class State(Enum):
    FUNCTION = "function"
    PARAMETER = "parameter"
    DECODE = "decode"
    FINAL = "final"

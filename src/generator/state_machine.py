from enum import Enum


class State(Enum):
    """
    This class defines the deferent state that
    will be used for my State-Machine.

    Attributes:
      - FUNCTION: It searches an encoded version of the function
      - PARAMETER: It searches an encoded version of the parameters
      - INCOMPLETE_PARAMETER: It searches an encoded version of the
                             parameters when they are not yet completed
      - DECODE: It decodes the encoded result of the previous state
      - FINAL: It returns the decoded and completed version of what
              was searched
    """
    FUNCTION = "function"
    PARAMETER = "parameter"
    INCOMPLETE_PARAMETER = "incomplete parameter"
    DECODE = "decode"
    FINAL = "final"

from pydantic import BaseModel
from typing import Any


class FunctionDefinition(BaseModel):
    """
    A model to verify the function file content that will
    be used by the LLM.

    Attributes:
      - name: the function's name
      - description: a description of the function
      - parameters: a dictionary containing the
      parameter's name and type
      - returns: a dictionary with the type of the
      return
    """
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


class InputPrompt(BaseModel):
    """
    A model to verify the input file content that will
    be used by the LLM.

    Attributes:
      - prompt: the user request in natural language
    """
    prompt: str

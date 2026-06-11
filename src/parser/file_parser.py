from src.parser.pydantic_class import InputPrompt, FunctionDefinition
from pydantic import ValidationError

import json
import os


def input_parser(file_path: str) -> None:
    """
    This function will verify if the file is properly formated
    before using it in the LLM.

    Parameter:
      - file_path: str

    Return
      -> None
    """
    try:
        with open(file_path, "r") as file:
            p_input = json.load(file)

    except PermissionError:
        raise ValueError("Permission denied!")

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON file is not properly formatted:\n {e}")

    for prompt in p_input:
        try:
            InputPrompt(**prompt)
        except ValidationError as e:
            raise ValueError(f"Invalid implementation: {e}")

    return None


def function_parser(file_path: str) -> None:
    """
    This function will verify if the file is properly formated
    before using it in the LLM.

    Parameter:
      - file_path: str

    Return
      -> None
    """
    try:
        with open(file_path, "r") as file:
            p_function = json.load(file)

    except PermissionError:
        raise ValueError("Permission denied!")

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON file is not properly formatted:\n {e}")

    for function in p_function:
        try:
            FunctionDefinition(**function)
        except ValidationError as e:
            raise ValueError(f"Invalid implementation: {e}")

    return None


def valid_input_check(file_path: str) -> None:
    """
    This function will check if the given input file is correctly
    formatted. If it isn't, the function raise an error that will
    be caught in the main program and stop it entirely.

    Parameter:
      - file: str

    Return
      -> str
    """
    path = "data/input/"

    if not os.path.isdir(path):
        raise ValueError("Directory does not exist.")

    if not os.path.isfile(file_path):
        raise ValueError(f"'{file_path}' path_file does not exist.")

    return None


def valid_output_check(file_path: str) -> None:
    """
    This function will check if the output folder is present
    and return a message informing that. If it isn't, the folder
    will be created and the function returns a message saying so.

    Parameter:
      - file_path: str

    Return
      -> str
    """
    path = "data/output/"

    if not os.path.isdir(path):
        os.mkdir(path)

    return None

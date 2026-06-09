import json
import os


def input_file_verification(file: str) -> str:
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
    func_def_file = "data/input/functions_definition.json"
    prompt_file = "data/input/function_calling_tests.json"

    try:
        if not os.path.isdir(path):
            raise ValueError("Directory does not exist.")

        if not os.path.isfile(func_def_file):
            raise ValueError(
                "'functions_definition.json' file does not exist."
                )

        if os.path.isfile(func_def_file):
            try:
                with open(func_def_file, "r") as f:
                    json.load(f)
                    f.close()

            except PermissionError:
                raise ValueError("Permission denied!")

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON file is not properly formatted:\n {e}"
                    )

        if not os.path.isfile(prompt_file):
            raise ValueError(
                "'function_calling_tests.json' file does not exist."
            )

        if os.path.isfile(prompt_file):
            try:
                with open(prompt_file, "r") as f:
                    json.load(f)
                    f.close()

            except PermissionError:
                raise ValueError("Permission denied!")

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON file is not properly formatted:\n {e}"
                )

        else:
            return "Input files and directory present!"

    except Exception as e:
        raise f"\033[1;31mUnexpected error!\n-> {e}\033[0m"
    return ""


def output_file_verification(file: str) -> str:
    """
    This function will check if the output folder is present
    and return a message informing that. If it isn't, the folder
    will be created and the function returns a message saying so.

    Parameter:
      - file: str

    Return
      -> str
    """
    path = "data/output/"

    if not os.path.isdir(path):
        os.mkdir(path)
        return "Output file and directory created!"
    return "Output file and directory present!"

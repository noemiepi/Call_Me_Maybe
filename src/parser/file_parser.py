import os
import json


def input_file_verification(file: str) -> str:
    path = "data/input/"
    func_def_file = "data/input/functions_definition.json"

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

        if not os.path.isfile(file):
            raise ValueError(
                "'function_calling_tests.json' file does not exist."
            )

        if os.path.isfile(f"{path}/{file}"):
            try:
                with open(f"{path}/{file}", "r") as f:
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
        return f"\033[1;31mUnexpected error!\n-> {e}\033[0m"
    return ""


def output_file_verification(file: str) -> str:
    path = "data/output/"

    if not os.path.isdir(path):
        os.mkdir(path)
        return "Output file and directory created!"
    return "Output file and directory present!"

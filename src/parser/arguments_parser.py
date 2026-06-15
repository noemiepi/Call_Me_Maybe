from src.parser.file_parser import valid_input_check, valid_output_check

import argparse
import os


def arg_parse() -> argparse.Namespace:
    """
    This function will create a few arguments the
    user can use when launching the program.

    Return
      -> argparse.Namespace
    """
    parser = argparse.ArgumentParser()

    count = 0
    for path in os.listdir("data/input/"):
        if os.path.isfile(os.path.join("data/input/", path)):
            count += 1

    if count < 2:
        raise ValueError("Missing input file(s)!")

    parser.add_argument("-f", "--functions_definition",
                        default="data/input/functions_definition.json",
                        choices=["data/input/functions_definition.json"],
                        type=valid_input_check,
                        help="Type the path to the function definition file: "
                        "'data/input/functions_definition.json'")

    parser.add_argument("-i", "--input",
                        default="data/input/function_calling_tests.json",
                        choices=["data/input/function_calling_tests.json"],
                        type=valid_input_check,
                        help="Type the path to the input file: "
                        "'data/input/function_calling_tests.json'")

    parser.add_argument("-o", "--output",
                        default="data/output/function_calling_results.json",
                        choices=["data/input/function_calling_results.json"],
                        type=valid_output_check,
                        help="Type the path to the output file: "
                        "'data/input/function_calling_results.json'")

    return parser.parse_args()

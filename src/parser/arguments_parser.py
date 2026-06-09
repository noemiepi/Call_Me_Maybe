from src.parser.file_parser import \
    input_file_verification, output_file_verification

import argparse


def arg_parse() -> argparse.Namespace:
    """
    This function will create a few arguments the
    user can use when launching the program.

    Return
      -> argparse.Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--functions_definition",
                        default="data/input/functions_definition.json",
                        choices=["data/input/functions_definition.json"],
                        metavar="path/file.json")

    parser.add_argument("-i", "--input",
                        default="data/input/function_calling_tests.json",
                        choices=["data/input/function_calling_tests.json"],
                        metavar="path/file.json",
                        type=input_file_verification)

    parser.add_argument("-o", "--output",
                        default="data/output/function_calling_results.json",
                        choices=["data/input/function_calling_results.json"],
                        metavar="path/file.json",
                        type=output_file_verification)

    return parser.parse_args()

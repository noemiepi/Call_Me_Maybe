from pydantic import BaseModel, PrivateAttr
from typing import Any, Pattern

import json
import re


class Vocabulary(BaseModel):
    """
    This class will will take every function from the input .json file
    and store them inside a class object dictionnary to be called.

    Attribute:
      - get_id_to_token_vocab(self, path: str) -> dict[int, str]
      - _create_function_list(self) -> list[dict[str, str]]
      - find_int_parameters(self, prompt: str) -> list[Any]:
      - find_str_parameters(self, prompt: str) -> list[Any]:
    """
    _vocab_list: dict[str, Any] = PrivateAttr()

    # Pattern
    _nb_pattern: Pattern[Any] = PrivateAttr()
    _quote_pattern: Pattern[Any] = PrivateAttr()

    def __init__(self) -> None:
        # Pattern
        self._nb_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        self._quote_pattern = re.compile(r"'[^']*'|\"[^\"]*\"")

    def get_id_to_token_vocab(self, path: str) -> dict[int, str]:
        """
        Creates a dictionnary that translate the tokens into functions.

        Parameter:
          - path: str

        Return
          -> dict[int, str]
        """
        try:
            with open(path, 'r') as f:
                vocab = json.load(f)

            rev_vocab = {}
            for key, value in vocab.items():
                rev_vocab[value] = key

            return rev_vocab
        except Exception as e:
            print("\033[1;31mError during the vocabulary's translation:\n"
                  f"-> {e}\033[0m")
            exit()

    def _create_function_list(self) -> dict[str, Any]:
        """
        By reading the .json file containing the functions,
        it will create a list isolating them.

        Return
          -> list[dict[str, Any]]
        """
        path: str = "data/input/"

        with open(f"{path}/functions_definition.json", 'r') as f:
            json_file: Any = json.load(f)

            self._vocab_list = json_file

        return self._vocab_list

    def find_int_parameters(self, prompt: str) -> list[Any]:
        """
        It does a first search through the prompt to
        find the type integer and number parameters.

        Parameter:
          - prompt: str

        Return
          -> list[Any]
        """
        return self._nb_pattern.findall(prompt)

    def find_str_parameters(self, prompt: str) -> Any:
        """
        It does a first search through the prompt to
        find the type string parameters

        Parameters:
          - prompt: str

        Return
          -> Any
        """
        # found_param: Any = ""
        candidates: list[Any] = []

        # Searches the quoted strings inside the prompt and makes a list
        quoted: list[Any] = self._quote_pattern.findall(prompt)

        for value in quoted:
            if value not in candidates:
                candidates.append(value)

        return candidates

        # if quoted:
        #     if len(quoted) > i:
        #         found_param = quoted[i]

        #     else:
        #         quoted[-1]

        # else:
        #     words = prompt.split()
        #     found_param = words[-1]

        # results = [found_param]
        # found_param = found_param.strip("?.,!'\"")
        # print(results)

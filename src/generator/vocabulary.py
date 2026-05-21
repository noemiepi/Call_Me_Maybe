from pydantic import BaseModel
from typing import Any

import json


class Vocabulary(BaseModel):
    """
    This class will will take every function from the input .json file
    and store them inside a class object dictionnary to be called.

    Attribute:
      - get_id_to_token_vocab(self, path: str) -> dict[int, str]
      - _create_function_list(self) -> dict[str, str]
    """
    vocab_list: list[str] = []

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

    def _create_function_list(self) -> list[str]:
        """
        By reading the .json file containing the functions,
        it will create a list isolating them.

        Return
          -> list[str]
        """
        path: str = "data/input/"

        with open(f"{path}/functions_definition.json", 'r') as f:
            json_file: Any = json.load(f)

            self.vocab_list = json_file

        return self.vocab_list

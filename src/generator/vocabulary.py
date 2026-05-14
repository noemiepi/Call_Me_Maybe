from pydantic import BaseModel
from typing import Any
import json


class Vocab(BaseModel):
    """
    Goal
    This class will will take every prompt from the input .json file
    and store them inside a class object list to be called.

    Attribute:
    - _create_function_list(self) -> list[str]
    """
    vocab_list: list[str] = []

    def _create_function_list(self) -> list[str]:
        """
        Goal
        By reading the .json file containing the functions,
        it will create a list isolating them.

        Parameter:
        - self

        Return
        -> list[str]
        """
        path: str = "data/input/"

        with open(f"{path}/functions_definition.json", 'r') as f:
            json_file: Any = json.load(f)

            for data in json_file:
                self.vocab_list.append(str(data["name"]))

        return self.vocab_list

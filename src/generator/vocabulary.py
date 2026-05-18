from pydantic import BaseModel
from typing import Any
import json


class Vocab(BaseModel):
    """
    This class will will take every function from the input .json file
    and store them inside a class object dictionnary to be called.

    Attribute:
      - _create_function_list(self) -> dict[str, str]
    """
    vocab_dict: dict[str, str] = {}

    def _create_function_list(self) -> dict[str, str]:
        """
        By reading the .json file containing the functions,
        it will create a list isolating them.

        Return
          -> dict[str, str]
        """
        path: str = "data/input/"

        with open(f"{path}/functions_definition.json", 'r') as f:
            json_file: Any = json.load(f)

            for data in json_file:
                self.vocab_dict.update({str(data["name"]) :
                                        str(data["parameters"])})

        return self.vocab_dict

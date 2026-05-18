from pydantic import BaseModel
from typing import Any
import json


class Prompt(BaseModel):
    """
    This class will will take every prompt from the input .json file
    and store them inside a class object list to be called.

    Attributes:
      - get_next_prompt(self) -> str | None
      - _create_prompt_list(self) -> None
    """
    prompt_list: list[str] = []

    def get_next_prompt(self) -> str | None:
        """
        Using the prompt list, which is a class object, it will
        take the first prompt off of the list and return it.

        Return
          -> str | None
        """
        if not self.prompt_list:
            return None

        return self.prompt_list.pop(0)

    def _create_prompt_list(self) -> None:
        """
        By reading the .json file containing the prompts,
        it will create a list isolating them.

        Return
          -> None
        """
        path: str = "data/input/"

        with open(f"{path}/function_calling_tests.json", 'r') as f:
            json_file: Any = json.load(f)

            for data in json_file:
                self.prompt_list.append(str(data["prompt"]))

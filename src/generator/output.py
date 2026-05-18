from pydantic import BaseModel
from typing import Any
import json


class Output(BaseModel):
    """
    This class will generate the output .json file with the prompt,
    the function's name and it's parameters.

    Attributes:
      - join_results(self, prompt: str, gen_output: dict[str, Any]) -> None
      - write_output(self) -> None
    """
    result: list[Any] = []

    def join_results(self, gen_output: dict[str, Any]) -> None:
        """
        It joins the LLM generation's results inside a dictionary and
        appends it to the result object which is a list,
        making its type list[dict[str, Any]].

        Parameter:
          - gen_output: dict[str, Any]

        Return
          -> None
        """
        self.result.append(gen_output)

    def write_output(self) -> None:
        """
        Taking the class object (result), it will write it's content inside
        a .json file that is either created if non-existent or overwritten.

        Return
          -> None
        """
        output_path = "data/output/function_calling_results.json"

        try:
            with open(output_path, 'w') as f:
                f.write(json.dumps(self.result, indent=4))

        except Exception as e:
            print("\033[1;31mAn error occured during the output writing:\n"
                  f"-> {e}\033[0m")

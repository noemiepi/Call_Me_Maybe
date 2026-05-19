from llm_sdk import Small_LLM_Model

from src.generator.vocabulary import Vocabulary

from pydantic import BaseModel, PrivateAttr
from typing import Any

import time


class Call_Me_Maybe(BaseModel):
    """
    This class will be used to find the wanted function
    and the right parameters for a given prompt.

    Attributes:
      - model_post_init(self, context: Any | None) -> None
      - answer_generation(self, prompt: str) -> dict[str, Any]
      - gen_function_name(self) -> str
      - gen_function_param(self, function: str) -> dict[str, Any]
    """
    _model: Small_LLM_Model = PrivateAttr()
    _vocab: Vocabulary = PrivateAttr()

    def model_post_init(self, context: Any | None) -> None:
        """
        It finishes to initialize the model
        with the necessary vocabulary.

        Parameter:
          - context: Any | None

        Return
          -> None
        """
        self._model = Small_LLM_Model()
        self._vocab = Vocabulary()

    def answer_generation(self, prompt: str) -> dict[str, Any]:
        """
        It reunites every prompt, function to solve it and
        its parameters found by the LLM and return it in
        a format that is easily translated to JSON format.

        Parameter:
          - prompt: str

        Return
          -> dict[str, Any]
        """
        start: float = time.time()
        # vocab_list: list[str] = self._vocab._create_function_list()
        output_result: dict[str, Any] = {}

        # Prompt
        output_result['prompt'] = prompt

        # Function name
        function_name: str = self.gen_function_name()
        output_result['name'] = function_name

        # Function parameters
        function_param: dict[str, Any] = self.gen_function_param(function_name)
        output_result['parameters'] = function_param

        end: float = time.time()

        # Visualization
        print(prompt)
        print(f">>> Function: {function_name}")
        print(f">>> Parameters: {function_param}")
        print(f"Generation time: \033[1;94m{(end-start)/60:.2f}s\033[0m")
        print()

        return output_result

    def gen_function_name(self) -> str:
        """
        It chooses the correct function name to solve
        the given prompt by passing through a State-Machine.

        Return
          -> str
        """
        # Prompt for the llm
        prompt: str = "Choose the best matching function that answers the " \
                "prompt.\n"

        # State of the machine
        state: str = ""

        # Passing through the State-Machine
        chosen_function = "bleh"

        return chosen_function

    def gen_function_param(self, function: str) -> dict[str, Any]:
        """
        It chooses the correct function's parameters
        and find them inside the prompt.

        Parameter:
          - function: str

        Return
          -> dict[str, Any]
        """
        # Prompt for the llm
        prompt: str = "Choose the best matching function that answers the " \
                "prompt.\n"

        # State of the machine
        state: str = ""

        # Passing through the State-Machine
        correct_parameters: dict[str, Any] = {"bla": "blu"}

        return correct_parameters

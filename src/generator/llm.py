from llm_sdk import Small_LLM_Model

from src.generator.vocabulary import Vocabulary
from src.generator.state_machine import State

from pydantic import BaseModel, PrivateAttr
from typing import Any

import numpy as np

import time


class Call_Me_Maybe(BaseModel):
    """
    This class will be used to find the wanted function
    and the right parameters for a given prompt.

    Attributes:
      - model_post_init(self, context: Any | None) -> None
      - generated_answer(self, prompt: str) -> dict[str, Any]
      - gen_function_name(self) -> str
      - gen_function_param(self, function: str) -> dict[str, Any]
      - generate(self, prompt: str, state: State, max_tokens: int) -> str
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

    def generated_answer(self, prompt: str) -> dict[str, Any]:
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
        output_result: dict[str, Any] = {}
        vocab_list: list[str] = self._vocab._create_function_list()

        # Prompt
        output_result['prompt'] = prompt

        # Function name
        function_name: str = self.gen_function_name(vocab_list)
        output_result['name'] = function_name

        # Function parameters
        function_param: dict[str, Any] = self.gen_function_param(function_name,
                                                                 vocab_list)
        output_result['parameters'] = function_param

        end: float = time.time()

        # Visualization
        print(prompt)
        print(f"-> Function: {function_name}")
        print(f"-> Parameters: {function_param}")
        print(f"Generation time: \033[1;94m{(end-start)/60:.2f}s\033[0m")
        print()

        return output_result

    def gen_function_name(self, vocab_list: list[str]) -> str:
        """
        It chooses the correct function name to solve
        the given prompt by passing through a State-Machine.

        Parameter:
          - vocab_list: list[str]

        Return
          -> str
        """
        # Prompt for the llm
        prompt: str = "Choose the best matching function in " \
                      f"{vocab_list} that answers the prompt.\n"

        # State of the machine
        state: State = State.FUNCTION

        # Passing through the State-Machine
        chosen_function = self.generate(prompt, vocab_list, state, 100)

        return chosen_function

    def gen_function_param(self, function: str,
                           vocab_list: list[str]) -> dict[str, Any]:
        """
        It chooses the correct function's parameters
        and find them inside the prompt.

        Parameters:
          - function: str
          - vocab_list: list[str]

        Return
          -> dict[str, Any]
        """
        # Prompt for the llm
        prompt: str = f"Choose the best matching function in {vocab_list} " \
                      "that answers the prompt.\n"

        # State of the machine
        state: State = ""

        # Passing through the State-Machine
        matching_parameters: dict[str, Any] = {"bla": "blu"}

        return matching_parameters

    def generate(self, prompt: str, vocab_list: list[str],
                 state: State, max_tokens: int) -> str:
        prompt_ids: list[int] = self._model.encode(prompt)[0].tolist()

        current_state: State = state
        current_token: list[int] = []

        gen_output: str = ""

        for _ in range(max_tokens):
            all_token: list[int] = prompt_ids + current_token

            if state == State.FINAL:
                return gen_output

            if state == State.FUNCTION:
                new_ids: list[int] = (
                        self._model.encode(gen_output)[0].tolist())
                prompt_ids.extend(new_ids)

            else:
                logits: list[float] = self._model.get_logits_from_input_ids(
                    all_token)

                valid_token: set[int] = set()
                # for func in vocab_list:

                sorted_token = sorted(logits, reverse=True)

                token = logits.index(sorted_token[0])

                return self._model.decode(token)

        return gen_output

from llm_sdk import Small_LLM_Model

from src.generator.vocabulary import Vocabulary
from src.generator.state_machine import State

from pydantic import BaseModel, PrivateAttr
from typing import Any, Union

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

    path: str = ""
    func_dict: list[str] = []
    vocab_dict: dict[int, str] = {}
    _functions_token: dict[str, list[int]] = {}

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

        self.path = self._model.get_path_to_vocab_file()
        self.func_dict = self._vocab._create_function_list()
        self.vocab_dict = self._vocab.get_id_to_token_vocab(self.path)

        for func in self.func_dict:
            ids: list[int] = self._model.encode(func['name'])[0].tolist()
            self._functions_token[func['name']] = ids

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

        # Prompt
        output_result['prompt'] = prompt
        print(prompt)

        # Function name
        function_name: str = self.gen_function_name(prompt)
        output_result['name'] = function_name
        print(f"-> Function: {function_name}")

        # Function parameters
        function_param: dict[str, Any] = self.gen_function_param(prompt,
                                                                 function_name)
        output_result['parameters'] = function_param

        end: float = time.time()

        # Visualization
        # print(f"-> Function: {function_name}")
        print(f"-> Parameters: {function_param}\n")
        sec = end-start
        if sec > 60:
            timer = time.strftime("%M.%S", time.gmtime(end-start))
            print(f"Generation time: \033[1;94m{timer}min\033[0m\n\n")
        else:
            timer = time.strftime("%S", time.gmtime(end-start))
            print(f"Generation time: \033[1;94m{timer}s\033[0m\n\n")

        return output_result

    def gen_function_name(self, prompt: str) -> str:
        """
        It chooses the correct function name to solve
        the given prompt by passing through a State-Machine.

        Parameter:
          - prompt: str
          - vocab_list: list[str]

        Return
          -> str
        """
        # Prompt for the llm
        llm_prompt: str = ("You are a helpful function finder who has access "
                           "to this list of functions:\n")

        for func in self.func_dict:
            llm_prompt += f"- {func['name']}: {func['description']}\n"

        llm_prompt += ("Which function needs to be called "
                       "to answer this prompt: "
                       f"{prompt} ?")

        # State of the machine
        state: State = State.FUNCTION

        # Passing through the State-Machine
        chosen_function: str = self.generate(llm_prompt, state, None)

        return chosen_function

    def gen_function_param(self, prompt: str, function: str) -> dict[str, Any]:
        """
        It chooses the correct function's parameters
        and find them inside the prompt.

        Parameters:
          - prompt: str
          - function: str

        Return
          -> dict[str, Any]
        """
        parameters: dict[str, Any] = {}
        # Prompt for the llm
        llm_prompt: str = ("You are a helpful parameters who has access "
                           "to this list of functions:\n")

        for func in self.func_dict:
            llm_prompt += f"- {func['name']}: {func['parameters']}\n"

        llm_prompt += (f"Find the parameters of {function} inside this"
                       f" {prompt} without solving it")

        # State of the machine
        state: State = State.PARAMETER

        # Passing through the State-Machine
        matching_parameters: str = self.generate(llm_prompt, state, function)

        return matching_parameters

    def generate(self, prompt: str, state: State, function: None | str) -> str:
        """
        It generates what is wanted (prompt) depending
        on the given state of the machine.

        Parameters:
          - prompt: str
          - state: State

        Return
          -> str
        """
        prompt_ids: list[int] = self._model.encode(prompt)[0].tolist()

        current_token: list[int] = []
        max_tokens: int = 100

        gen_output: str = ""

        for _ in range(max_tokens):
            all_token: list[int] = prompt_ids + current_token
            logits: list[float] = self._model.get_logits_from_input_ids(
                all_token)

            if state == State.FINAL:
                return gen_output

            if state == State.FUNCTION:
                valid_tokens: set[int] = set()
                for func in self.func_dict:
                    if func['name'].startswith(gen_output):
                        new_token: list[int] = self._functions_token[
                            func['name']]
                        next_pos: int = len(current_token)

                        if next_pos < len(new_token):
                            valid_tokens.add(new_token[next_pos])

                # Changes the machine's state
                state = State.DECODE

            if state == State.PARAMETER:
                for func in self.func_dict:

                    if func['name'] == function:

                        for param_name, param_info in \
                            func['parameters'].items():
                            param_type = param_info['type']

                            if param_type == "number":
                                valid_tokens: set[int] = set()

                            elif param_type == "string":
                                valid_tokens: set[int] = set()

                            elif param_type == "boolean":
                                valid_tokens: set[int] = set()

                            gen_output += f"{param_name}: {param_type}\n"

                state = State.DECODE
                print(f"\nParameter(s):\n{gen_output}")

            else:
                if not valid_tokens:
                    break

                logits_masked: np.NDArray[Any] = np.full_like(logits, -np.inf,
                                                              dtype=float)
                for token_id in valid_tokens:
                    logits_masked[token_id] = logits[token_id]

                # Select best token
                best_token_id: int = int(np.argmax(logits_masked))
                current_token.append(best_token_id)

                # Convert tokens into a string
                token_string = self.vocab_dict.get(best_token_id, "")
                token_string = (
                    token_string.replace('\u2581', '').replace('\u0120', '')
                )
                gen_output += token_string

                # Avoid infinite loop if token is empty
                if not token_string:
                    break

                # Change the states of the machine
                if any(func['name'] == gen_output for func in self.func_dict):
                    state = State.FINAL
                else:
                    state = State.FUNCTION

        return "outside loop"

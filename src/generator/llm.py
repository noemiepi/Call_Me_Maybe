from llm_sdk import Small_LLM_Model

from src.generator.vocabulary import Vocabulary
from src.generator.state_machine import State

from pydantic import BaseModel, PrivateAttr
from typing import Any

import numpy as np
import time
import re


class Call_Me_Maybe(BaseModel):
    """
    This class will be used to find the wanted function
    and the right parameters for a given prompt.

    Attributes:
      - model_post_init(self, context: Any | None) -> None
      - generated_answer(self, prompt: str) -> dict[str, Any]
      - gen_function_name(self) -> str
      - gen_function_param(self, function: str) -> dict[str, Any]
      - generate(self, prompt: str, state: State, function: None | str) -> str
      - find_parameters(self, param_type: str, prompt: str, loop: int) -> Any
    """
    _model: Small_LLM_Model = PrivateAttr()
    _vocab: Vocabulary = PrivateAttr()

    _number_pattern = PrivateAttr()

    path: str = ""
    func_dict: list[dict[str, Any]] = []
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
        self._number_pattern = re.compile(r"-?\d+(?:\.\d+)?")

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
        if len(function_param) == 1:
            print(f"-> Parameter: {function_param}\n")
        else:
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
        llm_prompt: str = ("You are a helpful assitant. Your role is to "
                           "extract the parameters to solve this function:\n")

        for func in self.func_dict:
            if func['name'] == function:
                llm_prompt += f"- {func['name']}: {func['parameters']}\n"

        llm_prompt += ("You will find the parameters of the function inside "
                       f"this prompt: {prompt}.\nDo NOT solve the prompt")

        # State of the Machine
        state: State = State.PARAMETER

        # Passing through the State-Machine
        matching_parameters: str = self.generate(llm_prompt, state, function)

        cleaned_parameters = matching_parameters.rstrip(",")

        for func in self.func_dict:
            if func['name'] == function:

                for para_name, para_info in func['parameters'].items():
                    param_type = para_info['type']

                    if param_type == "number" or param_type == "integer":
                        parameters = dict(item.split(": ") for item in
                                          cleaned_parameters.split(", "))

        if len(parameters) > 0:
            return parameters

        return matching_parameters

    def generate(self, prompt: str, state: State, function: None | str) -> str:
        """
        It generates what is wanted (prompt) depending
        on the given state of the machine.

        Parameters:
          - prompt: str
          - state: State
          - function: None | str

        Return
          -> str
        """
        prompt_ids: list[int] = self._model.encode(prompt)[0].tolist()

        current_token: list[int] = []
        max_tokens: int = 100
        loop: int = -1

        gen_output: str = ""

        for _ in range(max_tokens):
            all_token: list[int] = prompt_ids + current_token

            logits: list[float] = self._model.get_logits_from_input_ids(
                all_token)

            valid_tokens: set[int] = set()

            if state == State.FINAL:
                return gen_output

            if state == State.FUNCTION:
                for func in self.func_dict:
                    if func['name'].startswith(gen_output):
                        new_token: list[int] = self._functions_token[
                            func['name']]
                        next_pos: int = len(current_token)

                        if next_pos < len(new_token):
                            valid_tokens.add(new_token[next_pos])

                if not valid_tokens:
                    break

                logits_masked: np.NDArray[Any] = np.full_like(logits, -np.inf,
                                                              dtype=float)
                for token_id in valid_tokens:
                    logits_masked[token_id] = logits[token_id]

                best_token_id: int = int(np.argmax(logits_masked))

                # Changes the machine's state
                state = State.DECODE

            if state == State.PARAMETER:
                best_token_id: int = int(np.argmax(logits))

                for func in self.func_dict:
                    if func['name'] == function:

                        for para_name, para_info in func['parameters'].items():
                            loop += 1
                            param_type = para_info['type']

                            if param_type == "number" or \
                            param_type == "integer":
                                found_param = self.find_parameters(param_type,
                                                                   prompt,
                                                                   loop)

                                if gen_output == "":
                                    gen_output += f"{para_name}: {found_param}"

                                else:
                                    gen_output += (f", {para_name}: "
                                                   f"{found_param}")

                # Changes the machine's state
                state = State.DECODE

            else:
                # Select best token
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

                # Change the states of the machine during function search
                if not any(func['name'] == gen_output
                           for func in self.func_dict):
                    state = State.FUNCTION

                else:
                    state = State.FINAL

        return gen_output

    def find_parameters(self, param_type: str, prompt: str, loop: int) -> Any:
        """
        It searchs the wanted parameters directly inside the prompt
        depending on its type.

        Parameters:
          - param_type: str
          - prompt: str
          - loop: int

        Return
          -> Any
        """
        if param_type == "number" or param_type == "integer":
            result_list = self._number_pattern.findall(prompt)
            result = result_list[loop]

            if param_type == "number":
                try:
                    result = float(result)
                except ValueError:
                    pass

                if result.is_integer():
                    result = int(result)
                if result is None:
                    return 0.0

            if param_type == "integer":
                try:
                    result = int(result)
                except ValueError:
                    pass

                if result is None:
                    return 0

            return result

        # elif param_type == "string":
        #     return "str"

        # elif param_type == "boolean":
        #     return "bool"

        # return param_type

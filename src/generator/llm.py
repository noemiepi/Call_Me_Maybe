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
      - generate(self, prompt: str, state: State,
                 function: str | None = None) -> str:
    """
    # Class
    _model: Small_LLM_Model = PrivateAttr()
    _vocab: Vocabulary = PrivateAttr()

    # Vocabulary
    _path: str = PrivateAttr()
    _func_dict: list[dict[str, Any]] = PrivateAttr()
    _vocab_dict: dict[int, str] = PrivateAttr()
    _functions_token: dict[str, list[int]] = PrivateAttr()

    def model_post_init(self, context: Any | None) -> None:
        """
        It finishes to initialize the model
        with the necessary vocabulary.

        Parameter:
          - context: Any | None

        Return
          -> None
        """
        # Class
        self._model = Small_LLM_Model()
        self._vocab = Vocabulary()

        # Vocabulary
        self._path = self._model.get_path_to_vocab_file()
        self._func_dict = self._vocab._create_function_list()
        self._vocab_dict = self._vocab.get_id_to_token_vocab(self._path)
        self._clean_vocab = [
            (token, token.replace('\u2581', ' ').replace('\u0120', ' '), token_id)
            for token, token_id in self._vocab._vocab_list.items()
        ]
        self._functions_token = {}

        for func in self._func_dict:
            ids: list[int] = self._model.encode(func['name'])[0].tolist()
            self._functions_token[func['name']] = ids

    def generated_answer(self, prompt: str) -> dict[str, Any]:
        """
        It reunites every prompt, function and its parameters found
        by the LLM to solves the prompt and returns the answer in
        a format that is easily translated to JSON format.

        Parameter:
          - prompt: str

        Return
          -> dict[str, Any]
        """
        function_param: dict[str, Any] = {}
        output_result: dict[str, Any] = {}

        # Starts the timer
        start: float = time.time()

        # Prompt
        output_result['prompt'] = prompt
        print(f"{prompt}\n")

        # Prompt for the llm
        llm_prompt: str = ("You are a function caller assistant who has access"
                           " to this list of functions:\n")

        for func in self._func_dict:
            llm_prompt += (f"- {func['name']} ({func['parameters']}): "
                           f"{func['description']}\n")

        llm_prompt += ("Select the right function that needs to be called "
                       "and what arguments have to be used to answer this "
                       f"prompt: {prompt} ?")

    # ------ Function name ----------------------------------------------------

        # State of the machine
        state: State = State.FUNCTION

        # Passing through the State-Machine
        function: str = self.generate(llm_prompt, state)
        output_result['name'] = function

        # Visualization
        print(f"-> Function: {function}")

    # ------ Function parameters ----------------------------------------------

        # Reunite candidates
        int_candidates = self._vocab.find_int_parameters(prompt)
        str_candidates = self._vocab.find_str_parameters(prompt)

        # State of the Machine
        state: State = State.PARAMETER

        # Passing through the State-Machine
        matching_parameters: str = self.generate(llm_prompt, state, function,
                                                 int_candidates, str_candidates)

        # Putting the parameters inside a dictionnary instead of a string
        try:
            parameters = dict(item.split(": ", 1) for item in
                            matching_parameters.split(", "))

            if len(parameters) > 0:
                function_param = parameters

            else:
                raise ValueError

        except ValueError:
            function_param = matching_parameters

        output_result['parameters'] = function_param

        # Visualization
        if len(function_param) == 1:
            print(f"-> Parameter: {function_param}\n")

        else:
            print(f"-> Parameters: {function_param}\n")

        # Stops the timer
        end: float = time.time()

        sec = end-start

        if sec > 60:
            timer = time.strftime("%Mmin%S", time.gmtime(end-start))
            print(f"Generation time: \033[1;94m{timer}\033[0m")

        else:
            timer = time.strftime("%S", time.gmtime(end-start))
            print(f"Generation time: \033[1;94m{timer}s\033[0m")

        return output_result

    def generate(self, prompt: str, state: State,
                 function: str | None = None,
                 int_candidates: list[str] | None = None,
                 str_candidates: list[str] | None = None) -> str:
        """
        It generates what is wanted (prompt) depending
        on the given state of the machine.

        Parameters:
          - prompt: str
          - state: State
          - function: str | None = None

        Return
          -> str
        """
        prompt_ids: list[int] = self._model.encode(prompt)[0].tolist()

        valid_tokens: set[int] = set()
        current_token: list[int] = []
        max_tokens: int = 100

        gen_output: str = ""

        for _ in range(max_tokens):
            logits: list[float] = self._model.get_logits_from_input_ids(
                prompt_ids + current_token)

            # The generation is complete
            if state == State.FINAL:
                return gen_output

            # Generates an encoded version of the function
            if state == State.FUNCTION:
                for func in self._func_dict:
                    if func['name'].startswith(gen_output):
                        new_token: list[int] = self._functions_token[
                            func['name']]
                        next_pos: int = len(current_token)

                        if next_pos < len(new_token):
                            valid_tokens.add(new_token[next_pos])

                # Changes the machine's state
                state = State.DECODE

            if state == State.PARAMETER:
                if function is None:
                    return "pas fonction (je vais oublier de retirer ca)"

                for func in self._func_dict:
                    if func['name'] == function:
                        for p_name, p_type in func['parameters'].items():
                            par_type = p_type.get('type')
                            valid_tokens = self._valid_token(par_type,
                                                             int_candidates,
                                                             str_candidates)

                # Changes the machine's state
                state = State.DECODE

            # Decodes what was previously encoded
            if state == State.DECODE:
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
                token_string = self._vocab_dict.get(best_token_id, "")

                # Clean the string
                clean_string = (
                    token_string.replace('\u2581', ' ').replace('\u0120', ' ')
                )

                if not clean_string:
                    break

                gen_output += clean_string

                # Change the states of the machine during function search
                if not any(func['name'] == gen_output
                           for func in self._func_dict):
                    state = State.FUNCTION

                else:
                    state = State.FINAL

                # if output == "":
                #     output += (f"{param_name}: {found_param}")

                # else:
                #     output += (f", {param_name}: {found_param}")

                # return output

        return gen_output

    def _valid_token(self, par_type: str, int_candidates: list[str],
                     str_candidates: list[str]) -> None:
        token: list[Any] = []
        tmp: str = ""

        for token, clean, token_id in self._clean_vocab:
            if par_type in ("number", "integer"):
                print("int")
                candidate_value = tmp + clean
                for value in int_candidates:
                    if value.startswith(candidate_value):
                        token.append(token_id)
                        break

            if par_type == "string":
                print("str")

            if par_type == "boolean":
                print("bool")

        return token

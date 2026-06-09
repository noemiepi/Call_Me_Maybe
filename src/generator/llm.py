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
                 function: str | None = None) -> str
      - _valid_token(self, par_type: str, nb_cand: list[str] | None,
                     str_cand: list[str] | None, gen_output: str) -> list[int]
      - change_state(self, gen_output: str, function: str | None,
                     par_list: list[str], nb_cand: list[str] | None,
                     str_cand: list[str] | None,
                     state: State) -> tuple[State, Any]
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
            (token, token.replace('\u2581', ' ').replace('\u0120', ' '),
             token_id)
            for token_id, token in self._vocab_dict.items()
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
        if prompt == "" or prompt is None:
            output_result['prompt'] = prompt
            output_result['name'] = "No function"
            output_result['parameters'] = "No parameters"
            print("Empty prompt")
            return output_result
        output_result['prompt'] = prompt
        print(f"{prompt}\n")

        # Prompt for the llm
        llm_prompt = ("You are a function calling assistant \n"
        "Your task: given a user request, output a JSON object selecting\n"
        "the right function or fill its arguments.\n\n"
        "Examples:\n"
        "- User request: Replace all vowels in 'Programming is fun' "
        "with asterisks\n"
        "- Output: {'name': 'fn_substitute_string_with_regex', "
        "'parameters': {'source_string': 'Programming is fun', "
        "'regex': 'vowels', 'replacement': 'asterisks'}\n\n"
        "Available functions:\n")
        for func in self._func_dict:
            llm_prompt += f"- {func['name']}: {func['description']}"
            for p_name, p_type in func['parameters'].items():
                llm_prompt += (f"   {p_name} ({p_type['type']})")
        llm_prompt += f"\n\nUser request: {prompt}"

    # ------ Function name ---------------------------------------

        # State of the machine
        state: State = State.FUNCTION

        # Passing through the State-Machine
        function: str = self.generate(llm_prompt, state)
        output_result['name'] = function

        # Visualization
        print(f"-> Function: {function}")

    # ------ Function parameters ---------------------------------

        # Reunite candidates
        int_candidates = self._vocab.find_int_parameters(prompt)
        str_candidates = self._vocab.find_str_parameters(prompt)

        # State of the Machine
        state = State.PARAMETER

        # Passing through the State-Machine
        matching_parameters: str = self.generate(llm_prompt,
                                                 state,
                                                 function,
                                                 int_candidates,
                                                 str_candidates)

        # Putting the parameters inside a dictionnary instead of a string
        try:
            parameters = dict(item.split(": ", 1) for item in
                              matching_parameters.split(", "))

            if len(parameters) > 0:
                function_param = parameters

                output_result['parameters'] = function_param

            else:
                raise ValueError

        except ValueError:
            output_result['parameters'] = matching_parameters

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
                 nb_cand: list[str] = [],
                 str_cand: list[str] = []) -> str:
        """
        It generates what is wanted (prompt) depending
        on the given state of the machine.

        Parameters:
          - prompt: str
          - state: State
          - function: str | None = None
          - nb_cand: list[str] | None = None,
          - str_cand: list[str] | None = None

        Return
          -> str
        """
        prompt_ids: list[int] = self._model.encode(prompt)[0].tolist()

        valid_tokens: list[int] = []
        current_token: list[int] = []
        max_tokens: int = 150

        filled_par: str = ""
        new_par: str = ""
        par_type: str = ""
        par_list: list[str] = []
        par_time: int = 0

        gen_output: str = ""

        for _ in range(max_tokens):
            logits: list[float] = self._model.get_logits_from_input_ids(
                prompt_ids + current_token)

            # The generation is complete
            if state == State.FINAL:
                if filled_par:
                    return filled_par
                return gen_output

            # Generates an encoded version of the function
            if state == State.FUNCTION:
                for func in self._func_dict:
                    if func['name'].startswith(gen_output):
                        new_token: list[int] = self._functions_token[
                            func['name']]
                        next_pos: int = len(current_token)

                        if next_pos < len(new_token):
                            valid_tokens.append(new_token[next_pos])

                # Changes the machine's state
                state = State.DECODE

            if state == State.PARAMETER or state == State.INCOMPLETE_PARAMETER:
                if function is None:
                    return "No function provided"

                for func in self._func_dict:
                    if func['name'] == function:
                        if par_time == 0:
                            for p_name, p_type in func['parameters'].items():
                                if p_name not in par_list:
                                    par_list.append(p_name)
                                    for type, par_type in p_type.items():
                                        par_list.append(par_type)
                                par_time += 1

                        if par_list is not []:
                            for p_name, p_type in func['parameters'].items():
                                if p_type != par_type:
                                    par_type = p_type.get('type')
                                    valid_tokens = self._valid_token(
                                                                     par_type,
                                                                     nb_cand,
                                                                     str_cand,
                                                                     gen_output
                                                                    )

                # Changes the machine's state
                state = State.DECODE

            # Decodes what was previously encoded
            if state == State.DECODE:
                if not valid_tokens:
                    print("No valid token")
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
                    print("No clean string")
                    break

                gen_output += clean_string

                state, new_par, par_list, nb_cand, \
                    str_cand = self.change_state(gen_output, function,
                                                 par_list, nb_cand,
                                                 str_cand, state)

                # Accumulates the new found parameter and
                # resets the token and output
                if state == State.PARAMETER or state == State.FINAL:
                    if new_par:
                        print(f"new_par-> {new_par}")
                        filled_par += new_par
                        gen_output = ""
                        current_token = []

        return "Failed to generate an answer"

    def change_state(self, gen_output: str, function: str | None,
                     par_list: list[str], nb_cand: list[str],
                     str_cand: list[str],
                     state: State) -> tuple[State, Any, Any, Any, Any]:
        """
        Changes the state of the machine to either FUNCTION,
        PARAMETER or FINAL depending on if it found what is searched.

        Parameters:
          - gen_ouput: str
          - function: str
          - filled_par: dict[str, Any]
          - par_list: list[str]
          - nb_cand: list[str]
          - str_cand: list[str]

        Return
          -> tuple[State, Any, Any, Any, Any]
        """
        new_par: str = ""
        i_cand: int = 0
        i: int = 0

        # During the function search
        if function is None:
            if not any(func['name'] == gen_output for func in self._func_dict):
                return State.FUNCTION, None, None, None, None

            else:
                return State.FINAL, None, None, None, None

        # During the parameters search
        else:
            par_name = par_list[i]
            par_type = par_list[i + 1]

            new_par += f"{par_name}: "

            if par_type == "string":
                if str_cand is not None and gen_output not in str_cand:
                    state = State.INCOMPLETE_PARAMETER
                else:
                    if len(par_list) > 2:
                        new_par += f"{gen_output}, "
                        for candidate in str_cand:
                            if candidate == gen_output:
                                break
                            i_cand += 1
                        str_cand.pop(i_cand)
                        par_list.pop(0)
                        par_list.pop(0)
                        state = State.PARAMETER
                    else:
                        new_par += gen_output
                        state = State.FINAL

            if par_type == "integer":
                if nb_cand is not None and gen_output not in nb_cand:
                    state = State.PARAMETER
                else:
                    if len(par_list) > 2:
                        new_par += f"{int(gen_output)}, "
                        for candidate in str_cand:
                            if candidate == gen_output:
                                break
                            i_cand += 1
                        nb_cand.pop(i_cand)
                        par_list.pop(0)
                        par_list.pop(0)
                        state = State.PARAMETER
                    else:
                        new_par += f"{int(gen_output)}"
                        state = State.FINAL

            if par_type == "number":
                if nb_cand is not None and gen_output not in nb_cand:
                    state = State.INCOMPLETE_PARAMETER
                else:
                    if len(par_list) > 2:
                        new_par += f"{float(gen_output)}, "
                        for candidate in str_cand:
                            if candidate == gen_output:
                                break
                            i_cand += 1
                        nb_cand.pop(i_cand)
                        par_list.pop(0)
                        par_list.pop(0)
                        state = State.PARAMETER
                    else:
                        new_par += f"{float(gen_output)}"
                        state = State.FINAL

            if par_type == "boolean":
                state = State.FINAL

            return state, new_par, par_list, nb_cand, str_cand

    def _valid_token(self, par_type: str, nb_cand: list[str] | None,
                     str_cand: list[str] | None, gen_output: str) -> list[int]:
        """
        Select the necessary token to find find the parameters

        Parameters:
          - par_type: str
          - nb_cand: list[str] | None
          - str_cand: list[str] | None
          - gen_output: str

        Return
          -> list[Any]
        """
        token_lst: list[int] = []

        for token, clean, token_id in self._clean_vocab:

            if par_type == "string":
                if str_cand is None:
                    token_lst.append(token_id)
                    continue

                candidate_value = gen_output + clean
                for value in str_cand:
                    if value.startswith(candidate_value):
                        token_lst.append(token_id)
                        break

            elif par_type in ("number", "integer"):
                if nb_cand is None:
                    token_lst.append(token_id)
                    continue

                if clean == "":
                    continue

                candidate_value = gen_output + clean
                for value in nb_cand:
                    if value.startswith(candidate_value):
                        token_lst.append(token_id)
                        break

            elif par_type == "boolean":
                if clean in ("true", "false"):
                    token_lst.append(token_id)
                print("bool")

        return token_lst

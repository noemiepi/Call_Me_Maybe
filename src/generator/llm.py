from llm_sdk import Small_LLM_Model

from src.generator.vocabulary import Vocabulary
from src.generator.state_machine import State

from pydantic import BaseModel, PrivateAttr
from typing import Any, Pattern

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
      - find_parameters(self, param_name: str, param_type: str, prompt: str,
                        output: str, loop: int) -> Any
    """
    # Class
    _model: Small_LLM_Model = PrivateAttr()
    _vocab: Vocabulary = PrivateAttr()

    # Pattern
    _nb_pattern: Pattern[Any] = PrivateAttr()
    _quote_pattern: Pattern[Any] = PrivateAttr()
    _prompt_in: Pattern[Any] = PrivateAttr()
    _prompt_with: Pattern[Any] = PrivateAttr()
    _regex_keywords: dict[str, str] = {}

    # Vocabulary
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
        # Class
        self._model = Small_LLM_Model()
        self._vocab = Vocabulary()

        # Pattern
        self._nb_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        self._quote_pattern = re.compile(r"'[^']*'|\"[^\"]*\"")
        self._prompt_in = re.compile(r"\bin\s+('[^']*'|\"[^\"]*\")")
        self._prompt_with = re.compile(r"\bwith\s+('*[^']*'*|\"*[^\"]*\"*)")
        self._regex_keywords = ["letters", "vowels", "consonants",
                                "uppercases", "lowercases", "numbers",
                                "digits", "spaces", "whitespaces",
                                "punctuations"]

        # Vocabulary
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
        print(f"{prompt}\n")

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
            print(f"Generation time: \033[1;94m{timer}min\033[0m")
        else:
            timer = time.strftime("%S", time.gmtime(end-start))
            print(f"Generation time: \033[1;94m{timer}s\033[0m")

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
                       f"to answer this prompt: {prompt} ?")

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
        # llm_prompt: str = ("Task: You are a function selector. "
        #                    "Given a user request, extract the explicit values "
        #                    "from the user request to populate the parameters. "
        #                    "Do NOT solve the problem or calculate the answer."
        #                    " Only extract the arguments.")

        # for func in self.func_dict:
        #     if func['name'] == function:
        #         llm_prompt += f"{func['name']}:\n"

        #     for param_name, param_type in func['parameters'].items():
        #         llm_prompt += f"{param_name} ({param_type['type']})\n"

        # llm_prompt += f"User request: {prompt}."

        # State of the Machine
        state: State = State.PARAMETER

        # Passing through the State-Machine
        matching_parameters: str = self.generate(prompt, state, function)

        # Putting the parameters inside a dictionnary instead of a string
        try:
            parameters = dict(item.split(": ", 1) for item in
                            matching_parameters.split(", "))
        except ValueError:
            return matching_parameters

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
        i_int: int = -1
        i_str: int = -1

        gen_output: str = ""

        for _ in range(max_tokens):
            all_token: list[int] = prompt_ids + current_token

            logits: list[float] = self._model.get_logits_from_input_ids(
                all_token)

            best_token_id: int = int(np.argmax(logits))

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

                if not valid_tokens:
                    break

                logits_masked: np.NDArray[Any] = np.full_like(logits, -np.inf,
                                                              dtype=float)
                for token_id in valid_tokens:
                    logits_masked[token_id] = logits[token_id]

                best_token_id = int(np.argmax(logits_masked))

                # Changes the machine's state
                state = State.DECODE

            if state == State.PARAMETER:
                for func in self.func_dict:
                    if func['name'] == function:

                        for par_name, par_info in func['parameters'].items():
                            par_type = par_info['type']

                            # Find the int and float type of parameters
                            if par_type == "number" or par_type == "integer":
                                i_int += 1

                                gen_output = self.find_parameters(par_name,
                                                                  par_type,
                                                                  prompt,
                                                                  gen_output,
                                                                  i_int)

                                if len(func['parameters']) - 1 == i_int:
                                    return gen_output

                            if par_type == "string":
                                i_str += 1

                                gen_output = self.find_parameters(par_name,
                                                                  par_type,
                                                                  prompt,
                                                                  gen_output,
                                                                  i_str)

                                if len(func['parameters']) - 1 == i_str:
                                    return gen_output

                            if par_type == "boolean":
                                pass

                # Changes the machine's state
                state = State.DECODE

            else:
                # Select best token
                current_token.append(best_token_id)

                # Convert tokens into a string
                token_string = self.vocab_dict.get(best_token_id, "")

                if not token_string:
                    break

                # Clean the string
                token_string = (
                    token_string.replace('\u2581', ' ').replace('\u0120', ' ')
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

    def find_parameters(self, param_name: str, param_type: str, prompt: str,
                        output: str, i: int) -> Any:
        """
        It searchs the wanted parameters directly inside the prompt
        depending on its type.

        Parameters:
          - param_name: str
          - param_type: str
          - prompt: str
          - output: str
          - i: int

        Return
          -> Any
        """
        found_param: Any = ""

        if param_type == "number" or param_type == "integer":
            # Search the number(s) inside the prompt and makes a list
            results: list[Any] = self._nb_pattern.findall(prompt)
            found_param = results[i]

            # Try to convert the number into a float
            if param_type == "number":
                try:
                    found_param = float(found_param)
                except ValueError:
                    pass

                # Checks if there is a comma
                if found_param.is_integer():
                    found_param = int(found_param)

                if found_param is None:
                    found_param = 0.0

            # Try to convert the number into an int
            if param_type == "integer":
                try:
                    found_param = int(found_param)
                except ValueError:
                    pass

                if found_param is None:
                    found_param = 0

        if param_type == "string":
            # Search the strings inside the prompt and makes a list
            quoted: list[Any] = self._quote_pattern.findall(prompt)

            if param_name == "source_string":
                match_in: list[Any] = self._prompt_in.search(prompt)
                if match_in:
                    found_param = match_in.group(1)

            elif param_name == "replacement":
                match_rep: list[Any] = self._prompt_with.search(prompt)
                if match_rep:
                    found_param = match_rep.group(1)

            elif param_name == "regex":
                prompt_lower = prompt.lower()
                found_param = next(keyword for keyword in self._regex_keywords
                                   if keyword in prompt_lower)

            elif quoted:
                if len(quoted) > i:
                    found_param = quoted[i]

                else:
                    quoted[-1]

            else:
                words = prompt.split()
                found_param = words[-1]

            results = [found_param]
            found_param = found_param.strip("?.,!'\"")
            print(results)

        if output == "":
            output += (f"{param_name}: {found_param}")

        else:
            output += (f", {param_name}: {found_param}")

        return output

        # elif param_type == "boolean":
        #     return "bool"

from llm_sdk import Small_LLM_Model
from src.generator.vocabulary import Vocab
from pydantic import BaseModel, PrivateAttr
from typing import Any
import numpy
from numpy.typing import NDArray


class Call_Me_Maybe(BaseModel):
    """
    This class will be used to find the wanted function
    and the right parameters for a given prompt.

    Attributes:
      - model_post_init(self, context: Any | None) -> None
      - tokenization(self, prompt: str) -> str
      - generate_function_name(self, prompt: str) -> str
      - generate_parameters(self, function: str) -> dict[str, Any]
    """
    _model: Small_LLM_Model = PrivateAttr()
    _vocab: Vocab = PrivateAttr()

    def model_post_init(self, context: Any | None) -> None:
        self._model = Small_LLM_Model()
        self._vocab = Vocab()

        # self.func_name = ""
        # self.params = {}

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
        vocab_dict: dict[str, str] = self._vocab._create_function_list()
        output_result: dict[str, Any] = {}

        # Prompt
        output_result['prompt'] = prompt

        # Function name
        func_name: str = self.generate_function_name(prompt, vocab_dict)
        output_result['name'] = func_name

        # Function parameters
        func_param: dict[str, Any] = self.generate_parameters(prompt)
        output_result['parameters'] = func_param

        # Visualization
        print(prompt)
        print(f">>> Function: {func_name}")
        print(f">>> Parameters: {func_param}")
        print()

        return output_result

    def generate_function_name(self, prompt: str,
                               func_dict: dict[str, str]) -> str:
        """
        It chooses the correct function name to solve
        the given prompt within the list of functions.

        Parameters:
          - prompt: str
          - func_dict: dict[str, str]

        Return
          -> str
        """
        prompt_tensors: list[int] = self._model.encode(prompt)[0].tolist()

        # task = f"{prompt}"
        task = "Task: Respond to every prompt with 'a'"

        # task = "Your goal is to search the function in " \
        # f"{funcs} that corresponds to the given prompt"

        # task: dict[str, str] = func_dict
        # task = f"\nResolve the following prompt: {prompt}\n"

        logits = self._model.get_logits_from_input_ids(prompt_tensors)

        sorted_token = sorted(logits, reverse=True)

        token = logits.index(sorted_token[0])

        return self._model.decode(token)

    def generate_parameters(self, prompt: str, func: str) -> dict[str, Any]:
        """
        It chooses the correct function's parameters.

        Parameter:
          - prompt: str
          - func: str

        Return
          -> dict[str, Any]
        """
        return "function not build"
        # task = "Your goal is to find the parameters of " \
        # f"{self.func_name} inside the prompt"

        # if self.func_name == "fn_add_numbers":
        #     '"parameters": {"a": "number", "b": "number"}'

        # elif self.func_name == "fn_greet":
        #     '"parameters": {"name": "string"}'

        # elif self.func_name == "fn_reverse_string":
        #     '"parameters": {"s": "string"}'

        # elif self.func_name == "fn_get_square_root":
        #     '"parameters": {"a": "number"}'

        # elif self.func_name == "fn_substitute_string_with_regex":
        #     '"parameters": {"source_string": "string", "regex": "string",'
        #     '"replacement": "string"}'

        # return self.params

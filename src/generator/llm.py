from llm_sdk import Small_LLM_Model
from src.generator.vocabulary import Vocab
from pydantic import BaseModel, PrivateAttr
from typing import Any


class My_LLM(BaseModel):
    """
    Goal
    This class will be used to find the wanted function
    and the right parameters for a given prompt.

    Attributes:
    - model_post_init(self, context: Any | None) -> None
    - tokenization(self, llm: str, prompt: str) -> str
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

    def tokenization(self, prompt: str) -> str:
        """
        Goal

        Parameters:
        - self
        - prompt: str

        Return
        -> str
        """
        funcs = self._vocab.vocab_list
        # path = "data/input/functions_definition.json"

        self._instruction = "Your goal is to search the function in " \
        f"{funcs} that corresponds to the given prompt"

        # instruction = funcs
        # instruction += f"Resolve the following prompt: {prompt}"

        tensors = self._model.encode(prompt)

        probabilities = self._model.get_logits_from_input_ids(tensors.tolist()[0])

        sorted_token = sorted(probabilities)

        token = probabilities.index(sorted_token[0])

        return self._model.decode(token)

    def generate_function_name(self, prompt: str,
                               function_list: list[str]) -> str:
        """
        Goal

        Parameters:
        - self
        - prompt: str
        - function_list: list[str]

        Return
        -> str
        """
        self._instruction = "Your goal is to search the function " \
        "that corresponds to the given prompt"

        self.func_name += "fn_"

        return self.func_name

    def generate_parameters(self, prompt: str) -> dict[str, Any]:
        """
        Goal

        Parameters:
        - self
        - prompt: str

        Return
        -> dict[str, Any]
        """
        self._instruction = "Your goal is to find the parameters of " \
        f"{self.func_name} inside the prompt"

        if self.func_name == "fn_add_numbers":
            '"parameters": {"a": "number", "b": "number"}'

        elif self.func_name == "fn_greet":
            '"parameters": {"name": "string"}'

        elif self.func_name == "fn_reverse_string":
            '"parameters": {"s": "string"}'

        elif self.func_name == "fn_get_square_root":
            '"parameters": {"a": "number"}'

        elif self.func_name == "fn_substitute_string_with_regex":
            '"parameters": {"source_string": "string", "regex": "string",'
            '"replacement": "string"}'

        return self.params

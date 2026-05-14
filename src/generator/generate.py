from src.generator.llm import My_LLM


def answer_generation(prompt: str) -> None:
    """
    Goal

    Parameter
    - prompt: str

    Return
    -> None
    """
    model = My_LLM()

    print(prompt)
    print(model.tokenization(prompt))
    print()

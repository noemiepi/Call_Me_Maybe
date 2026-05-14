from src.parser.arguments_parser import arg_parse
from src.generator.generate import answer_generation
from src.generator.prompt import Prompt
from src.generator.llm import My_LLM
from src.generator.output import Output


if __name__ == "__main__":
    # try:
        # Parsing
        try:
            arg = arg_parse()

        except ValueError as e:
            print("\033[1;31mError!\033[0m")
            print(f"\033[1;31m-> {e}\033[0m")
            exit()

        except Exception as e:
            print("\033[1;31mAn unexpected error occured during the parsing:\n"
                  f"-> {e}\033[0m")
            exit()

        # Model loading
        model = My_LLM()
        model.model_post_init(None)

        # Prompt gathering
        prompt_creator = Prompt()
        prompt_creator._create_prompt_list()

        # Output creation
        output = Output()

        # Prompt processing
        while True:
            next_prompt = prompt_creator.get_next_prompt()

            if next_prompt is None:
                break

            answer_generation(next_prompt)
            output.join_results(next_prompt, "test", {"param1": "test",
                                                      "param2": "testbis"})

        output.write_output()

        print("Generation finished!")

    # except Exception as e:
    #     print("\033[1;31mAn unexpected error occured:\n"
    #           f"-> {e}\033[0m")
    #     exit()

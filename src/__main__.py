from src.parser.arguments_parser import arg_parse
from src.generator.prompt import Prompt
from src.generator.llm import Call_Me_Maybe
from src.generator.output import Output
# from src.generator.vocabulary import Vocabulary

from typing import Any

import time


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
        model = Call_Me_Maybe()
        model.model_post_init(None)

        # Prompt gathering
        prompt_creator = Prompt()
        prompt_creator._create_prompt_list()

        # Output creation
        output = Output()

        # Prompt processing
        start = time.time()
        while True:
            next_prompt = prompt_creator.get_next_prompt()

            if next_prompt is None:
                break

            gen_output: dict[str, Any] = model.generated_answer(next_prompt)
            output.join_results(gen_output)

        end = time.time()
        output.write_output()

        # voc = Vocabulary()
        # voc._create_function_list()
        # print(voc.vocab_dict)

        print("All generations completed!")
        print(f"Total time: \033[1;94m{(end-start)/60:.2f}s\033[0m")
    # except Exception as e:
    #     print("\033[1;31mAn unexpected error occured:\n"
    #           f"-> {e}\033[0m")
    #     exit()

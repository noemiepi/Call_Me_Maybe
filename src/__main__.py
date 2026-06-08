from src.parser.arguments_parser import arg_parse
from src.generator.prompt import Prompt
from src.generator.llm import Call_Me_Maybe
from src.generator.output import Output

from typing import Any

import time
import os


if __name__ == "__main__":
    # try:
        # Parsing arguments
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

        # Clears the terminal
        os.system('clear')

        # Visualizer

        print("\033[1;39mBut here's my number, so")
        print("  ____       _ _  ___  ___      ___  ___            _         "
              "   _         \033[5;39m|/\033[0m")
        print("\033[1;39m / __ \     | | | |  \/  |      |  \/  |           |"
              " |          | |     __i \033[5;39m-\033[0m")
        print("\033[1;39m| /  \/ __ _| | | | .  . | ___  | .  . | __ _ _   _|"
              " |__   ___  | |    |---|")
        print("| |    / _` | | | | |\/| |/ _ \ | |\/| |/ _` | | | | '_ \ / _ "
              "\ | |    |\033[1;47mhi!\033[0m|")
        print("| \__/\ (_| | | | | |  | |  __/ | |  | | (_| | |_| | |_) |  __"
              "/ |_|    |\033[1;37m:::\033[0m|")
        print("\033[1;39m \____/\__,_|_|_| \_|  |_/\___| \_|  |_/\__,_|\__, |"
              "_.__/ \___| (_)    |\033[1;37m:::\033[0m|")
        print("\033[1;39m                                               __/ |"
              "                   `\   \ ")
        print("                                              |___/           "
              "           \_=_\ \033[0m")

        # Model loading
        model: Call_Me_Maybe = Call_Me_Maybe()
        model.model_post_init(None)

        # Prompt gathering
        prompt_creator: Prompt = Prompt()
        prompt_creator._create_prompt_list()

        # Output creation
        output: Output = Output()

        # Prompt processing
        print("\n\033[1;39mStarting generation...\033[0m")

        # Starts the timer
        start: float = time.time()

        i: int = 1

        # Loop to answer every prompt
        while True:
            next_prompt: str | None = prompt_creator.get_next_prompt()

            if next_prompt is None:
                break

            print()
            print(f"|Prompt n°{i}|".center(79, "-"))
            print()

            gen_output: dict[str, Any] = model.generated_answer(next_prompt)
            output.join_results(gen_output)

            i += 1

        # Stops the timer
        end: float = time.time()

        output.write_output()

        print()
        print("".center(79, "-"))
        print()

        print("\033[1;39mEvery prompt has been answered!\033[0m")

        # Prints the timer
        sec: float = end-start
        timer: str = ""

        if sec > 60:
            timer = time.strftime("%Mmin%S", time.gmtime(end-start))
            print(f"Total time: \033[1;94m{timer}\033[0m")

        else:
            timer = time.strftime("%S", time.gmtime(end-start))
            print(f"Total time: \033[1;94m{timer}s\033[0m")

        # Visualizer
        print()
        print("\033[1;39mBefore you came into my life,".center(86))
        print("I missed you so bad".center(79))
        print("And you should know that,".center(79))
        print("So call me maybe\033[0m".center(83))
    # except Exception as e:
    #     print("\033[1;31mAn unexpected error occured:\n"
    #           f"-> {e}\033[0m")
    #     exit()

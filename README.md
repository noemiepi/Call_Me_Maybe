<div align="center">
    <i>This project has been created as part of the 42 curriculum by npillet</i>
    <h1>Call Me Maybe</h1>
    <h3>Introduction to function calling in LLMs</h3>
</div>

## Description
A **Large Language Models** (**LLM**) is a type of AI program that is capable of generating text. They are trained on large data bases and generate their answers to the user prommpt using this knowledge.

For this project, we will build a function calling program. It will reunite the name of the function needed to answer the prompt and extract the parameters from it to put it inside a valid JSON file.

The challenge here will be the use of a small language model (0.6B). These models are notoriously unreliable at generated a structure output, they suceed about 30% of the time to generate a valid JSON.<br>
Our objective is to have a 99%+ reliability with that same kind of model.<br>
We can achieve this with **constrained decoding**. It is a technique that guides the model's output token-by-token to garantee a valid structure, without relying solely on prompting.

## Instructions

```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

``` bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
```

``` bash
uv run python -m src
```

## Explanation and justification

### Algorithm explanation
Describe your constrained decoding approach in detail

### Design decisions
Explain key choices in your implementation

### Performance analysis
Discuss accuracy, speed, and reliability of your solution

### Challenges faced
Document difficulties encountered and how you solved them

### Testing strategy
Describe how you validated your implementation

### Example usage
Provide clear examples of running your program




## Resources

### Notions:

#### Parsing
- https://docs.python.org/3/library/argparse.html

- https://www.tutorialspoint.com/article/How-to-change-the-permission-of-a-file-using-Python

#### JSON files
- https://docs.python.org/3/library/json.html#encoders-and-decoders

#### AI Model
- https://huggingface.co/Qwen/Qwen3-0.6B

#### LLM State-Machine
- https://github.com/MLConvexAI/LLM-State-Machine

#### REGEX
- https://medium.com/@victoriousjvictor/understanding-regular-expressions-regex-e1c048f5aa6c

### Github:
- [Overtekk](https://github.com/Overtekk/Call_Me_Maybe)

- [Flipsbone](https://github.com/Flipsbone/Call-Me-Maybe-42)

- [naha7777](https://github.com/naha7777/Call_Me_Maybe)

- [fcaval42](https://github.com/fcaval42/CallMeMaybe)

- [saeedehAsheri](https://github.com/saeedehAsheri/CallMeMaybe)

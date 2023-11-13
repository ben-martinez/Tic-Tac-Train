import argparse
from openai import OpenAI
import time
from tqdm import tqdm

client = OpenAI(
    api_key='',
)

INTRO_MESSAGE = {'role' : 'user',
                  'content' : """
                            You will be given an example game sequence by the user, where there are 2 players and they make alternating moves
                            with player 1 making the first move. Please derive a logical rule that governs the game
                            (less than or equal to 30 words) that explains why the game ended in that state.
                            """
                 }

def generate_rules(input_filename, output_filename, num_examples, model, query_interval):
    # Set up the context for the rest of the prompts
    prompts = []

    # GPT wil be prompted to generate one rule per example in the training dataset
    with open(input_filename, 'r') as training_dataset:
        training_examples = training_dataset.readlines()
        for i in range(min(num_examples, len(training_examples))):
            prompt = {
                "role": "user",
                "content": training_examples[i]
            }
            prompts.append(prompt)

    # Query GPT with prompts and save rules
    with open(output_filename, 'w', encoding='utf-8') as file:
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]

            # Get response
            response_dict = client.chat.completions.create(messages=[INTRO_MESSAGE, prompt], model=model, max_tokens=100)
            response = response_dict.choices[0].message.content

            # Write response
            file.write(response + '\n')

            # Pause to obey rate limit
            time.sleep(query_interval)


def main():
    parser = argparse.ArgumentParser(description='Generates rules for a file of example games by querying GPT')
    parser.add_argument('--input-file', type=str, required=True, help='File of game examples')
    parser.add_argument('--output-file', type=str, required=True, help='File to write rules')
    parser.add_argument('--query-interval', type=float, default=0.0, help='Interval in seconds to query GPT')
    parser.add_argument('--num-examples', type=int, default=100, help='Number of examples')
    parser.add_argument('--model', type=str, default='gpt-4', help='GPT model to use')
    args = parser.parse_args()

    generate_rules(args.input_file, args.output_file, args.num_examples, args.model, args.query_interval)


if __name__ == '__main__':
    main()

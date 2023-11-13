import argparse
from openai import OpenAI
import time
import csv
from tqdm import tqdm

client = OpenAI(
    api_key='',
)

INTRO_MESSAGE = {'role' : 'user',
                  'content' : """
                            Rules of Game: \n
                            \n
                            1. Game Board Representation \n 
                            We are playing tic-tac-toe on a 3x3 grid. We can represent this grid using a matrix where c_{i, j} 
                            represents the cell in the i-th row and j-th column. Each cell can take one of three values: \n
                            0 for an empty cell, \n
                            1 for a cell marked by Player 1 (X), \n
                            -1 for a cell marked by Player 2 (O) \n
                            \n
                            2. Player Moves \n
                            Players alternate turns to mark an empty cell with their symbol. This can be represented as a function \n
                            1, if player = Player 1 and c_{i, j} = 0 \n
                            -1 if player = Player 2 and c_{i, j} = 0 \n
                            Invalid, otherwise \n
                            \n
                            3. Winning conditions \n
                            A player wins if they have three of their marks in a row, column, or diagonal. This can be represented as: \n
                            - Rows: \n
                            c_{1,1} + c_{1,2} + c_{1,3} = +/- 3 \n
                            c_{2,1} + c_{2,2} + c_{2,3} = +/- 3 \n
                            c_{3,1} + c_{3,2} + c_{3,3} = +/- 3 \n
                            - Columns: \n
                            c_{1,1} + c_{2,1} + c_{3,1} = +/- 3 \n
                            c_{1,2} + c_{2,2} + c_{3,2} = +/- 3 \n
                            c_{1,3} + c_{2,3} + c_{3,3} = +/- 3 \n
                            - Diagonals: \n
                            c_{1,1} + c_{2,2} + c_{3,3} = +/- 3 \n
                            c_{1,3} + c_{2,2} + c_{3,1} = +/- 3 \n
                            \n
                            4. Near-Win States \n
                            A near-win state is when a player has two marks in a line and the third cell is empty. This can be represented as: \n
                            - Rows: \n
                            |c_{1,1} + c_{1,2} + c_{1,3}| = 2 and one of c_{1,1}, c_{1,2}, c_{1,3} is 0 \n
                            Similar for other rows \n
                            - Columns: \n
                            |c_{1,1} + c_{2,1} + c_{3,1}| = 2 and one of c_{1,1}, c_{2,1}, c_{3,1} is 0 \n
                            Similar for other columns \n
                            - Diagonals: \n
                            |c_{1,1} + c_{2,2} + c_{3,3}| = 2 and one of c_{1,1}, c_{2,2}, c_{3,3} is 0 \n
                            |c_{1,3} + c_{2,2} + c_{3,1}| = 2 and one of c_{1,3}, c_{2,2}, c_{3,1} is 0 \n
                            \n
                            5. Priority of Moves \n
                            To play optimally, if it is your turn and there is a near-win state for you (defined above), 
                            play the move that will mark the third cell in the near-win. If it is your turn and you have 
                            no move that creates a winning condition, but your opponent has a near-win condition, block 
                            their win by playing your piece to where they would play to get a win condition. Try to create 
                            a win as fast as possible. \n
                            I will give you sequences of marks on the matrix and ask you for the optimal move expressed 
                            in the form (i, j). Player 1 plays the first mark, player 2 plays the second mark and they 
                            alternate turns. Before your move, you must clearly show the game in the notation that you 
                            provided. In particular, show EVERY term in the summations for EVERY row, column, and 
                            diagonal equation and for EVERY important state (win states, near-win states). Also reason 
                            with rule 5. Priority of moves explicitly by seeing if you have a winning move and playing it. 
                            Leverage the results of your evaluation to play the game optimally. You may not use code. 
                            Do you understand?
                            """
                 }
def generate_response(input_filename, output_filename, num_examples, model, query_interval):
    # Set up the context for the rest of the prompts
    prompts = []
    # GPT wil be prompted to generate the optimal move per test example
    with open(input_filename, 'r') as test_set:
        reader = csv.reader(test_set)
        rows = list(reader)[1:]
        for i in range(min(len(rows), num_examples)):
            # print(row)
            prompt = {
                "role": "user",
                "content": rows[i][0]
            }
            prompts.append(prompt)
    print(prompts)
    # Query GPT with prompts and save rules
    # expects a .csv file
    with open(output_filename, 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "response"])
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            # Get response
            response_dict = client.chat.completions.create(messages=[INTRO_MESSAGE, prompt], model=model, max_tokens=1000)
            response = response_dict.choices[0].message.content
            # Write response
            writer.writerow([prompt, response])
            # Pause to obey rate limit
            time.sleep(query_interval)

def main():
    parser = argparse.ArgumentParser(description='Generates rules for a file of example games by querying GPT')
    parser.add_argument('--input-file', type=str, required=True, help='File of game examples')
    parser.add_argument('--output-file', type=str, required=True, help='File to write rules')
    parser.add_argument('--query-interval', type=float, default=0.0, help='Interval in seconds to query GPT')
    parser.add_argument('--num-examples', type=int, required=True, default=100, help='Number of examples')
    parser.add_argument('--model', type=str, default='gpt-4', help='GPT model to use')
    args = parser.parse_args()
    generate_response(args.input_file, args.output_file, args.num_examples, args.model, args.query_interval)
if __name__ == '__main__':
    main()

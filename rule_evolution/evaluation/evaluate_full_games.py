from typing import List, Any, Tuple
from enum import Enum, auto
import logging
import asyncio
import random
from tqdm import tqdm

import sys
import re
sys.path.append('../util')
sys.path.append('../generation')

from util import fill_tags, get_basic_logger
import gpt_util
from game_util import game_sequence_to_latex_state, location_to_sequence_number, MinimaxAgent
import game_util
from evaluation import Evaluator, EvalResultType

api_key = None

# --- LOGGING ---
logger = get_basic_logger('evaluate_full_games.py')
logger.setLevel(logging.DEBUG)

# --- MOVE EVALUATION STATES ---
class FullGameEvaluator(Evaluator):
    def __init__(self,
                 client: gpt_util.LLMClient,
                 rule_library_context_template_filename: str,
                 game_example_template_filename: str,
                 minimax_cache_filename: str = None):
        super().__init__(client, 
                         rule_library_context_template_filename, game_example_template_filename, 
                         minimax_cache_filename)


    def get_best_moves(self, game_sequence): 
        best_moves, utility = self.agent.get_best_next_moves(game_sequence)
        return best_moves
    
    
    def get_utility(self, game_sequence): 
        best_moves, utility = self.agent.get_best_next_moves(game_sequence)
        return utility 


    def is_game_ended(self, game_sequence): 
        # returns 1 if game has reached a terminal state
        # check win
        game_state, next_marker = game_util.game_sequence_to_state(game_sequence)
        someone_has_won = game_util.is_winner_from_last_move(game_sequence, game_state)

        # check draw
        absolute_utility = self.get_utility(game_sequence)
        drawn = absolute_utility == 0 and len(game_sequence) == 9

        return someone_has_won or drawn
    

    async def evaluate_full_game(self, 
                                rule_library: str, 
                                markers: List[Any], 
                                model: str, 
                                dim: Tuple[int, int] = (3, 3), 
                                num_markers_in_win : int = 3,
                                debug: bool = False) -> Tuple[int, int]:
        """
        For all game sequences, given a rule library, plays a game against minimax and returns the result of game.

        Parameters:
            rule_library: A text description of rules to guide game strategy.
            markers: A length-3 list of markers of any type in this order: [empty, player_1, player_2]
            game_sequences: Lists of numbers in range 1...width * height representing the game moves.
            model: String identifier of the GPT model to ping.
            dim: A tuple that defines the dimension of the game board (width, height).
            num_markers_in_win: Number of consecutive markers to win the game.
            debug: Logs all requests, responses, best moves, and GPT moves if True.

        Returns:
            length_of_game: number of moves played in game (until l and excluding suboptimal moves
            result: 1 if GPT won, -1 if GPT lost and 0 if there was a draw

        Metric we are interested in: 
            length of game before loss 
        """
        game_sequence = []
        # 0 if the minimax agent plays the first move, 1 if GPT plays the first move
        first_player = random.randint(0, 1)
        current_player = first_player
        length_of_game = 0

        while True:
            if current_player == 0: 
                best_moves = self.get_best_moves(game_sequence)
                # randomly pick one of the best moves 
                chosen_move = random.choice(best_moves)
                game_sequence.append(chosen_move)
                current_player = 1
            else: 
                while True: 
                # prompt GPT using rule library and extract the move it generates 
                    evalresult = await self.evaluate_next_move(rule_library,
                                                    markers,
                                                    game_sequence,
                                                    model,
                                                    dim=dim,
                                                    num_markers_in_win=num_markers_in_win,
                                                    debug=debug)
                    if evalresult.type != EvalResultType.MOVE_NOT_FOUND: 
                        break
                move = evalresult.gpt_move

                # STOP DA GPT (replaces a square that's already played, plays something outside of board)
                if move < 1 or move > 9 or move in game_sequence:
                    return tuple((length_of_game, -1))
                game_sequence.append(move)
                current_player = 0
            length_of_game += 1

            # check if game has ended 
            absolute_utility = self.get_utility(game_sequence)
            relative_utility = self.agent.get_relative_utility(game_sequence, absolute_utility)
            if relative_utility < 0: 
                return tuple((length_of_game, -1))

            if len(game_sequence) == dim[0] ** 2: 
                return tuple((length_of_game, 1))
        
        return tuple((-1, -1))


    async def evaluate_n_full_games(self, 
                                rule_library: str, 
                                markers: List[Any], 
                                model: str, 
                                dim: Tuple[int, int] = (3, 3), 
                                num_markers_in_win : int = 3,
                                debug: bool = False, 
                                n: int = 1) -> Tuple[float, float]:
        """
        For all game sequences, given a rule library, plays n games against minimax and returns the result of game.

        Parameters:
            rule_library: A text description of rules to guide game strategy.
            markers: A length-3 list of markers of any type in this order: [empty, player_1, player_2]
            game_sequences: Lists of numbers in range 1...width * height representing the game moves.
            model: String identifier of the GPT model to ping.
            dim: A tuple that defines the dimension of the game board (width, height).
            num_markers_in_win: Number of consecutive markers to win the game.
            debug: Logs all requests, responses, best moves, and GPT moves if True.
            n: number of full games that should be played 

        Returns:
            average_length_of_game: average number of moves played in game over n games
            accuracy: # of games drawn / n 

        Metric we are interested in: 
            length of game before loss 
        """
        accuracy = 0 
        ave_length_of_game = 0

        print("Num games: ", n)
        print("Model:", model)
        print("Now running tests...")

        requests = []
        for _ in range(n):
            requests.append(self.evaluate_full_game(rule_library, markers, model))
        results = await asyncio.gather(*requests)

        for result in results:
            assert result[0] != -1

            if result[1] == 1: 
                accuracy += 1 
            ave_length_of_game += result[0]

        accuracy /= n
        ave_length_of_game /= n
        print("Results:", results)
        print("Accuracy:", accuracy)
        print("Average length of game:", ave_length_of_game)

        return tuple((ave_length_of_game, accuracy))
    

def main():
    with open('gpt3_5_5x10x5_BIG_rule_lib.txt', 'r') as file:
        # Read the contents of the file
        rule_lib = file.read()
    # print(rule_lib)
    with gpt_util.LLMClient(llm_type=gpt_util.LLMType.OPENAI, api_secret_key=api_key) as client:
        evaluator = FullGameEvaluator(client,
                                    'prompts/evaluation_rule_library.txt',
                                    'prompts/evaluation_example.txt')
        # 5x5x5 BAG GPT 4 (63%)
        # asyncio.run(evaluator.evaluate_n_full_games("Based on the provided suboptimal moves and optimal moves, here are the improved rules for playing tic-tac-toe:\n\nRule 1 (Unchanged):\nIf you have two in a row and the third square in that row is open, place your mark in the third square to win.\n\nRule 2 (Unchanged):\nBlock your opponent's win if they have two in a row and the third square in that row is open.\n\nRule 3 (Improved):\nIf the center square is open, place your marker there unless the corner square opposite to your opponent’s mark is open. However, if there are two open corners, the corner squares should take precedence to occupy instead of the central one.\n\nRule 4 (Improved): \nAlways prioritize threats in corners over sides. If your opponent has marked in a corner-square, and you have two open opposite corners, it’s optimal to take an opposite corner rather than placing a mark in the center.\n\nRule 5 (Improved):\nWhen you have the opportunity to create a setup for a two-in-a-row condition but there isn't a direct win condition, prioritize this over blocking one of your opponent's potential strides towards a two-in-a-row condition.\n\nRule 6 (Improved):\nIf you have a choice between a corner and a non-corner square, always choose the corner. Corners are involved in more winning lines (diagonal enabled) and therefore give you a higher probability of winning.\n\nRule 7 (New):\nWhen all corners are occupied and you have no direct wins or blocks, it is important to choose the side middle squares instead of the central one.\n\nRule 8 (New):\nIf the board state is open and you have the first move, choose a corner for setting up multiple winning opportunities.\n\nRule 9 (New):\nIf your opponent takes a non-central, non-corner square as their first move, it is optimal to take the center square. \n\nRule 10 (New):\nWhen no other rules apply, play the square that is involved in the highest number of unblocked lines still open for play. This generally prioritizes center > corner > edge. This rule has been generalized for larger than 3x3 boards.", [' ', 'X', 'O'], 'gpt-4', n=100))
        # asyncio.run(evaluator.evaluate_n_full_games("No rules.", [' ', 'X', 'O'], 'gpt-3.5-turbo', n=100))
        asyncio.run(evaluator.evaluate_n_full_games(rule_lib, [' ', 'X', 'O'], 'gpt-4', n=100))

if __name__ == '__main__':
    main()
from dataclasses import dataclass
from typing import List, Any, Tuple
from enum import Enum, auto
import logging
import asyncio

import sys
import re
sys.path.append('../util')
sys.path.append('../generation')

from util import fill_tags, get_basic_logger
from gpt_util import LLMClient
from game_util import game_sequence_to_latex_state, location_to_sequence_number, MinimaxAgent
from generation import Generator

api_key = None

# --- LOGGING ---
logger = get_basic_logger('evaluation.py')
logger.setLevel(logging.DEBUG)

# --- TAGS ---
RULE_LIBRARY_TAG = '<rule-library>'
GAME_STATE_TAG = '<game-state>'
CURRENT_PLAYER_TAG = '<current-player-marker>'

EMPTY_MARKER_TAG = '<empty-marker>'
PLAYER1_MARKER_TAG = '<player1-marker>'
PLAYER2_MARKER_TAG = '<player2-marker>'

# --- MOVE EVALUATION STATES ---
class EvalResultType(Enum):
    MOVE_NOT_FOUND = auto()
    LOSING_START = auto()
    SUBOPTIMAL_MOVE = auto()
    BEST_MOVE = auto()


@dataclass
class EvalResult:
    type: EvalResultType
    game_sequence: List[int]
    gpt_move: int
    best_moves: List[int]


@dataclass
class EvalResultBatch:
    formatting_accuracy: float
    best_move_accuracy: float
    eval_results: List[EvalResult]


class Evaluator:
    def __init__(self, 
                 client: LLMClient, 
                 rule_library_context_template_filename: str, 
                 game_example_template_filename: str,
                 minimax_cache_filename: str = None):
        """
        Constructs an Evaluator object.

        Parameters:
            client: A GPTClient for pinging.
            rule_library_context_template_filename: The filename for the file that contains the surrounding context for the rule library.
              File must include tags <rule-library>, <empty-marker>, <player1-marker>, <player2-marker>.
            game_example_template_filename: The filename for the file that contains the surrounding context for a game example.
              Should express that the next move must be in format <row, col>.  
              File must include tags <game-state> and <current-player-marker>.
            minimax_cache_file: Cache file for the minimax agent used in evaluation. Must be relative 
        """
        self.client = client
        
        # Rule library context template
        self.rule_library_context_template = f'Rule library: "{RULE_LIBRARY_TAG}". Empty: {EMPTY_MARKER_TAG}, Player 1: {PLAYER1_MARKER_TAG}, Player 2: {PLAYER2_MARKER_TAG}.'
        with open(rule_library_context_template_filename, 'r') as rule_library_context_template_file:
            self.rule_library_context_template = rule_library_context_template_file.read()
        assert self.rule_library_context_template.find(RULE_LIBRARY_TAG) != -1
        assert self.rule_library_context_template.find(EMPTY_MARKER_TAG) != -1
        assert self.rule_library_context_template.find(PLAYER1_MARKER_TAG) != -1
        assert self.rule_library_context_template.find(PLAYER2_MARKER_TAG) != -1

        # Game example template
        self.game_example_template = f'Current game state: {GAME_STATE_TAG}. Current player: {CURRENT_PLAYER_TAG}'
        with open(game_example_template_filename, 'r') as game_example_template_file:
            self.game_example_template = game_example_template_file.read()
        assert self.game_example_template.find(GAME_STATE_TAG) != -1
        assert self.game_example_template.find(CURRENT_PLAYER_TAG) != -1

        # Construct Minimax agent and load cache
        self.agent = MinimaxAgent(minimax_cache_in_filename=minimax_cache_filename)


    async def evaluate_next_move(self, 
                                rule_library: str, 
                                markers: List[Any], 
                                game_sequence: List[int],
                                model: str, 
                                dim: Tuple[int, int] = (3, 3), 
                                num_markers_in_win : int = 3,
                                debug: bool = False) -> EvalResult:
        """
        Given a rule library and game sequence, evaluates a model's next move against minimax.

        Parameters:
            rule_library: A text description of rules to guide game strategy.
            markers: A length-3 list of markers of any type in this order: [empty, player_1, player_2]
            game_sequence: A list of numbers in range 1...width * height representing the game moves.
            model: String identifier of the GPT model to ping.
            dim: A tuple that defines the dimension of the game board (width, height).
            num_markers_in_win: Number of consecutive markers to win the game.
            debug: Logs request, response, best moves, and GPT move if True.

        Returns:
            A tuple of (EvalResult comparing GPT's move response to minimax's best moves, list of best_moves).
        """
        if debug:
            logger.debug('--- DEBUG EVALUATION EXAMPLE ---')

        # State is already losing
        best_moves, utility = self.agent.get_best_next_moves(game_sequence, dim=dim, num_markers_in_win=num_markers_in_win)
        if self.agent.get_relative_utility(game_sequence, utility) < 0: # Disadvantageous utility 
            return EvalResult(EvalResultType.LOSING_START, game_sequence, None, best_moves)

        assert len(markers) == 3
        empty_marker, player1_marker, player2_marker = markers

        # Rule library context
        intro = fill_tags(self.rule_library_context_template, {
            RULE_LIBRARY_TAG : rule_library, 
            EMPTY_MARKER_TAG : str(empty_marker),
            PLAYER1_MARKER_TAG : str(player1_marker), 
            PLAYER2_MARKER_TAG : str(player2_marker)
        })

        # Game example
        game_text_state, current_player = game_sequence_to_latex_state(game_sequence, dim=dim, markers=markers)
        example = fill_tags(self.game_example_template, {
            GAME_STATE_TAG : game_text_state, 
            CURRENT_PLAYER_TAG : str(current_player)
        })

        if debug:
            logger.debug(f'---MESSAGE: {intro}\n{example}')

        # Get GPT response
        response = await self.client.get_one_shot_completion('user', intro, example, model)
        if debug:
            logger.debug(f'---RESPONSE: {response}')

        # Parse response
        pattern = r'@(\d),\s?(\d)@'
        move_matches = list(re.finditer(pattern, response))
        if len(move_matches) != 1: # if there is not one and only one pattern match then return
            return EvalResult(EvalResultType.MOVE_NOT_FOUND, game_sequence, None, best_moves)

        move_match = move_matches[0]
        move_str = move_match.group().replace(' ', '')

        row, col = int(move_str[1]) - 1, int(move_str[3]) - 1 # 1-index to 0-index
        sequence_num = location_to_sequence_number((row, col), dim=dim)

        if debug:
            logger.debug(f'---BEST MOVES: {best_moves}')
            logger.debug(f'---GPT MOVE: {sequence_num}')

        # GPT must play optimal move
        if sequence_num not in best_moves:
            return EvalResult(EvalResultType.SUBOPTIMAL_MOVE, game_sequence, sequence_num, best_moves)
        
        return EvalResult(EvalResultType.BEST_MOVE, game_sequence, sequence_num, best_moves)
    

    async def evaluate_accuracy(self, 
                                rule_library: str, 
                                markers: List[Any], 
                                game_sequences: List[List[int]],
                                model: str, 
                                dim: Tuple[int, int] = (3, 3), 
                                num_markers_in_win : int = 3,
                                debug: bool = False) -> EvalResultBatch:
        """
        For all game sequences, given a rule library, evaluates a model's next move against minimax.
        Returns two metrics - formatting accuracy and best move accuracy.

        Parameters:
            rule_library: A text description of rules to guide game strategy.
            markers: A length-3 list of markers of any type in this order: [empty, player_1, player_2]
            game_sequences: Lists of numbers in range 1...width * height representing the game moves.
            model: String identifier of the GPT model to ping.
            dim: A tuple that defines the dimension of the game board (width, height).
            num_markers_in_win: Number of consecutive markers to win the game.
            debug: Logs all requests, responses, best moves, and GPT moves if True.

        Returns:
            Formatting accuracy as a float i.e. out of useful starting game sequences,
                how many responses contained a correctly-formatted move?

            Best move accuracy as a float i.e. out of correctly formatted responses,
                how many responses gave the best move?

            List of tuple pairs containing the failed sequence as as the first value and the list of
            best moves as the second value.
        """
        assert len(game_sequences) > 0

        # Parallel eval requests (I/O bound)
        requests = []
        for sequence in game_sequences:
            requests.append(self.evaluate_next_move(rule_library,
                                                    markers,
                                                    sequence,
                                                    model,
                                                    dim=dim,
                                                    num_markers_in_win=num_markers_in_win,
                                                    debug=debug))
        eval_results: List[EvalResult] = await asyncio.gather(*requests)

        # Count up valid starts, valid outputs, and best moves
        valid_start_count = valid_output_count = best_move_count = 0
        for i, eval_result in enumerate(eval_results):
            if eval_result.type != EvalResultType.LOSING_START:
                valid_start_count += 1
                if eval_result.type != EvalResultType.MOVE_NOT_FOUND:
                    valid_output_count += 1
                    if eval_result.type == EvalResultType.BEST_MOVE:
                        best_move_count += 1

        # Calculate accuracies
        valid_output_accuracy = best_move_accuracy = 0.0
        if valid_start_count != 0:
            valid_output_accuracy = valid_output_count / valid_start_count
        if valid_output_count != 0:
            best_move_accuracy = best_move_count / valid_output_count

        return EvalResultBatch(valid_output_accuracy, best_move_accuracy, eval_results)
        
"""
with GPTClient(api_key) as client:   
    evaluator = Evaluator(client, 
                        'prompts/evaluation_rule_library.txt', 
                        'prompts/evaluation_example.txt',
                        minimax_cache_filename='caches/minimax_cache_dim-4,4_n-4.pkll')
    
    valid_output_accuracy, best_move_accuracy, _ = asyncio.run(evaluator.evaluate_accuracy('No rules', [0, 1, -1], [[3, 5, 7]] * 500, 'gpt-3.5-turbo', debug=True))
    print(valid_output_accuracy, best_move_accuracy)
"""
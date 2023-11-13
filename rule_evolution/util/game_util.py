from typing import List, Tuple, Any
from enum import Enum
from tqdm import tqdm
import pickle
import atexit
import random
import argparse
import os
import logging

import util

# --- LOGGING ---
logger = util.get_basic_logger('game_util.py')
logger.setLevel(logging.DEBUG)

# --- GAME SEQUENCES & STATES ---
class Marker(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2


def game_sequence_to_state(game_sequence: List[int], 
                           dim: Tuple[int, int] = (3, 3)) -> Tuple[List[List[int]], int]:
    """
    Converts a sequence of moves into a 2D state.
    Does not guard against terminal states.

    Parameters:
        game_sequence: A list of numbers in range 1...width * height representing the game moves.
        dim: A tuple that defines the dimension of the game board (width, height).

    Returns:
        Game state as a 2D array where 0 represents an empty cell,
        1 represents a move by player 1, and 2 represents a move by
        player 2.

        Next player marker.
    """
    assert len(dim) == 2

    width, height = dim
    assert width > 0 and height > 0

    assert all(isinstance(move, int) and 1 <= move <= width * height for move in game_sequence)

    state = [[Marker.EMPTY.value for _ in range(width)] for _ in range(height)]
    for i, move in enumerate(game_sequence):
        marker = i % 2 + 1
        row, col = (move - 1) // width, (move - 1) % width

        assert state[row][col] == Marker.EMPTY.value
        state[row][col] = marker

    next_marker = len(game_sequence) % 2 + 1
    return state, next_marker


def game_sequence_to_latex_state(game_sequence: List[int], 
                                 dim: Tuple[int, int] = (3, 3),
                                 markers: List[Any] = [0, 1, -1]) -> str:
    """
    Converts a sequence of moves into a LaTeX matrix state.
    Does not guard against terminal states.

    Parameters:
        game_sequence: A list of numbers in range 1...width * height representing the game moves.
        dim: A tuple that defines the dimension of the game board (width, height).
        markers: A list of markers for empty, player 1, player 2, respectively.

    Returns:
        Game state as a string LaTeX matrix with the specified markers in each cell.

        Next player marker.
    """
    assert len(markers) == 3

    state, next_marker_idx = game_sequence_to_state(game_sequence, dim=dim)

    text_state = '\\begin{bmatrix}\n'
    for row in range(len(state)):
        for col in range(len(state[row])):
            text_state += str(markers[state[row][col]])
            text_state += ' & ' if col < len(state[row]) - 1 else ' \\\\\n'
    text_state += '\\end{bmatrix}'

    return text_state, markers[next_marker_idx]


def location_to_sequence_number(location: Tuple[int, int],
                             dim: Tuple[int, int] = (3, 3)) -> int:
    """
    Converts a grid location into a sequence number used for game sequences.

    Parameters:
        location: A two-length tuple representing (row, col).
        dim: A tuple that defines the dimension of the game board (width, height).

    Returns:
        The sequence number as an int.
    """
    assert len(dim) == 2
    assert len(location) == 2

    width, _ = dim
    row, col = location

    return row * width + col + 1


def sequence_number_to_location(sequence_number: int, dim = (3, 3)) -> Tuple[int, int]:
    """
    Converts a sequence number to a grid location.

    Parameters:
        sequence_number: A number in range 1...width * height representing a game move.
        dim: A tuple that defines the dimension of the game board (width, height).
    
    Returns:
        The grid location tuple (row, col).
    """
    assert len(dim) == 2

    width, _ = dim
    return (sequence_number - 1) // width, (sequence_number - 1) % width


# --- GAME PROPERTIES ---
def move_in_bounds(game_state: List[List[int]], 
                   location: Tuple[int, int]) -> bool:
    """
    Returns whether the move (represented as a grid location) is in bounds.
    
    Parameters:
        game_state: Game state as a 2D array where 0 represents an empty cell, 1 represents a move by player 1, 
                    and 2 represents a move by player 2. 
        location: A two-length tuple representing (row, col). Assumed to be zero-indexed.

    Returns:
        True if the location is within the game state, False otherwise.
    """
    assert len(location) == 2

    row, col = location
    return row >= 0 and col >= 0 and row < len(game_state) and col < len(game_state[row])


def is_winner_from_last_move(game_sequence: List[int], 
                             game_state: List[List[int]], 
                             dim: Tuple[int, int] = (3, 3), 
                             num_markers_in_win: int = 3) -> bool:
    """
    Returns whether the last move caused a win on the game board.

    Parameters:
        game_sequence: A list of numbers in range 1...width * height representing the game moves.
        game_state: Game state as a 2D array where 0 represents an empty cell, 1 represents a move by player 1, 
                    and 2 represents a move by player 2. 
        dim: A tuple that defines the dimension of the game board (width, height).
        num_markers_in_win: Number of consecutive markers to win the game.

    Returns:
        True if the last move was a win, False otherwise.
    """
    row, col = sequence_number_to_location(game_sequence[-1], dim=dim)
    marker = game_state[row][col]

    directions = [[-1, -1], [-1, 0], [-1, 1], [0, 1]]
    for direction in directions:
        count = 1

        forward_row, forward_col = row + direction[0], col + direction[1]
        while move_in_bounds(game_state, (forward_row, forward_col)) and game_state[forward_row][forward_col] == marker:
            count += 1
            forward_row, forward_col = forward_row + direction[0], forward_col + direction[1]

        backward_row, backward_col = row - direction[0], col - direction[1]
        while move_in_bounds(game_state, (backward_row, backward_col)) and game_state[backward_row][backward_col] == marker:
            count += 1
            backward_row, backward_col = backward_row - direction[0], backward_col - direction[1]

        if count >= num_markers_in_win:
            return True
        
    return False


# --- MINIMAX EVALUATION ---
class MinimaxAgent:
    def __init__(self,
                 minimax_cache_in_filename : str = None,
                 minimax_cache_out_filename : str = None):
        """
        Constructs a GameAnalyzer object.

        Parameters:
            minimax_cache_in_filename: The name of the file that the cache for minimax should be loaded from.
            minimax_cache_out_filename: The name of the file that the cache for minimax should be saved to upon program termination.
        """
        # Load minimax cache using pickle
        self.minimax_cache = {}
        if minimax_cache_in_filename != None:
            if os.path.exists(minimax_cache_in_filename):
                with open(minimax_cache_in_filename, 'rb') as minimax_cache_file:
                    self.minimax_cache = pickle.load(minimax_cache_file)
            else:
                logger.warning(f'Could not open minimax cache path {minimax_cache_in_filename}!')
                
        # Save minimax cache at exit
        def save_minimax_cache():
            if minimax_cache_out_filename != None:
                dirname = os.path.dirname(minimax_cache_out_filename)
                if dirname != '':
                    os.makedirs(dirname, exist_ok=True)
                with open(minimax_cache_out_filename, 'wb') as minimax_cache_file:
                    pickle.dump(self.minimax_cache, minimax_cache_file)
        atexit.register(save_minimax_cache)


    def get_best_next_moves(self,
                            game_sequence: List[int], 
                            game_state: List[List[int]] = None, 
                            discount: float = 0.99,
                            dim: Tuple[int, int] = (3, 3),
                            num_markers_in_win: int = 3,
                            show_progress: bool = False) -> Tuple[List[int], float]:
        """
        Minimax to find the best next moves based on the game sequence.

        Parameters:
            game_sequence: A list of numbers in range 1...width * height representing the game moves.
            game_state: Game state as a 2D array where 0 represents an empty cell, 1 represents a move by player 1, 
                        and 2 represents a move by player 2. This parameter will be generated from the
                        'game_sequence' if it is not provided.
            dicount: For discouraging longer games - should win as soon as possible.
            dim: A tuple that defines the dimension of the game board (width, height).
            num_markers_in_win: Number of consecutive markers to win the game.
            show_progress: Show a progress bar as the function completes.

        Returns:
            A list of best moves as sequence numbers.

            A float representing the utility of any of the best moves.
        """
        # Load game state or build upon previous
        if game_state == None:
            game_state, next_marker = game_sequence_to_state(game_sequence, dim=dim)
        else:
            next_marker = len(game_sequence) % 2 + 1

        # Retrieve utility from cache if available
        key = (dim, num_markers_in_win, discount, tuple(tuple(row) for row in game_state))
        if key in self.minimax_cache:
            return self.minimax_cache[key]

        # Unpack dimensions
        assert len(dim) == 2
        width, height = dim
        
        # Check for win state
        if len(game_sequence) > 0 and is_winner_from_last_move(game_sequence, game_state, dim=dim, num_markers_in_win=num_markers_in_win):
            move_row, move_col = sequence_number_to_location(game_sequence[-1], dim=dim)
            utility = [0, 1, -1][game_state[move_row][move_col]]
            self.minimax_cache[key] = ([], utility)
            return self.minimax_cache[key]

        # Check for draw state
        possible_moves = list(set(range(1, width * height + 1)).difference(set(game_sequence)))
        if len(possible_moves) == 0:
            self.minimax_cache[key] = ([], 0)
            return self.minimax_cache[key]
        
        # Collect utilities
        best_utility = -float('inf') if next_marker == Marker.PLAYER1.value else float('inf')
        move_utilities = []
        for i in tqdm(range(len(possible_moves))) if show_progress else range(len(possible_moves)):
            # Choose 
            move = possible_moves[i]
            move_row, move_col = sequence_number_to_location(move, dim=dim)
            game_state[move_row][move_col] = next_marker

            # Explore
            _, utility = self.get_best_next_moves(game_sequence + [move], 
                                                game_state=game_state,
                                                discount=discount,
                                                dim=dim,
                                                num_markers_in_win=num_markers_in_win)
            utility *= discount
            move_utilities.append((move, utility))

            # Unchoose
            game_state[move_row][move_col] = Marker.EMPTY.value

            # Choose best utility
            if next_marker == Marker.PLAYER1.value:
                best_utility = max(best_utility, utility)
            else:
                best_utility = min(best_utility, utility)

        # Best moves
        best_moves = [move_utility[0] for move_utility in move_utilities if move_utility[1] == best_utility]

        # Cache and return the best moves and their utility
        self.minimax_cache[key] = (best_moves, best_utility)
        return self.minimax_cache[key]
    

    def get_relative_utility(self, game_sequence : List[int], absolute_utility : float):
        return [0, 1, -1][len(game_sequence) % 2 + 1] * absolute_utility


# --- PROGRAM FUNCTIONS ---
def play_minimax_agent(dim: Tuple[int, int] = (3, 3),
                       num_markers_in_win: int = 3,
                       cache_filename: str = None):
    """
    Interactive program to play against the Minimax agent.

    Parameters:
        dim: A tuple that defines the dimension of the game board (width, height).
        num_markers_in_win: Number of consecutive markers to win the game.
        cache_filename: File path to minimax cache to use for the game.
    """
    print('### PLAYING AGAINST MINIMAX AGENT ###')
    agent = MinimaxAgent(minimax_cache_in_filename=cache_filename)

    utility_to_ending = {1: 'You won!', -1 : 'You lost!', 0: 'Draw!'}
    game_sequence = []
    turn_num = 1

    while True:
        # Player's turn
        print(f'--- Turn {turn_num} --- ')
        print(f'Current game sequence: {game_sequence}')
        next_move = int(input('Your move: '))
        game_sequence.append(next_move)

        # Terminal state check after player's turn
        best_moves, utility = agent.get_best_next_moves(game_sequence, dim=dim, num_markers_in_win=num_markers_in_win)
        print(best_moves, utility)
        if len(best_moves) == 0:
            print(f'\n\n{utility_to_ending[utility]}')
            break

        # End turn
        turn_num += 1
        print('\n')

        # Agent's turn
        print(f'--- Turn {turn_num} --- ')
        print(f'Current game sequence: {game_sequence}')
        agent_move = random.choice(best_moves)
        print(f'Agent plays {agent_move}.')
        game_sequence.append(agent_move)

        # Terminal state check after agent's turn
        best_moves, utility = agent.get_best_next_moves(game_sequence, dim=dim, num_markers_in_win=num_markers_in_win)
        print(best_moves, utility)
        if len(best_moves) == 0:
            print(f'\n\n{utility_to_ending[utility]}')
            break

        # End turn
        turn_num +=1
        print('\n')


def cache_minimax_results(cache_filename: str,
                          dim: Tuple[int, int] = (3, 3),
                          num_markers_in_win: int = 3):
   """
   Computes and caches minimax results for a game configuration.

   Parameters:
        cache_filename: File path to load and save cache.
        dim: A tuple that defines the dimension of the game board (width, height).
        num_markers_in_win: Number of consecutive markers to win the game.
   """
   agent = MinimaxAgent(minimax_cache_in_filename=cache_filename, minimax_cache_out_filename=cache_filename)
   print(f'Starting computation for minimax cache on game configuration...')
   agent.get_best_next_moves([], dim=dim, num_markers_in_win=num_markers_in_win)
   print(f'Ending function to save cache results...')


def main():
    parser = argparse.ArgumentParser(description='Game Util Configuration')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subcommand for playing against Minimax agent
    parser_play = subparsers.add_parser('play', help='Play Tic-Tac-Toe against minimax agent')
    parser_play.add_argument('--dim', type=int, nargs=2, help='Dimensions of the board (width, height)', required=True)
    parser_play.add_argument('--num-markers-in-win', type=int, help='Number of consecutive markers to win', required=True)
    parser_play.add_argument('--cache-file', type=str, help='File path for cache', required=False)


    # Subcommand for caching Minimax results
    parser_cache = subparsers.add_parser('cache', help='Cache Minimax results')
    parser_cache.add_argument('--dim', type=int, nargs=2, help='Dimensions of the board (width, height)', required=True)
    parser_cache.add_argument('--num-markers-in-win', type=int, help='Number of consecutive markers to win', required=True)
    parser_cache.add_argument('--cache-file', type=str, help='File path for cache', required=True)


    args = parser.parse_args()

    if args.command == 'play':
        play_minimax_agent(dim=tuple(args.dim), num_markers_in_win=args.num_markers_in_win, cache_filename=args.cache_file)
    elif args.command == 'cache':
        cache_minimax_results(args.cache_file, dim=tuple(args.dim), num_markers_in_win=args.num_markers_in_win)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
from typing import List, Tuple, Dict, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass
from collections import defaultdict
import random
import sys
sys.path.append('../util')
from game_util import MinimaxAgent

import time

class GeneratorMarker(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = -1


class MarkerSequenceType(Enum):
    WIN = auto(),
    NEAR_WIN = auto(),
    NEAR_FORK_WIN = auto(),
    NEAR_FORK_LOSS = auto(),
    NEAR_LOSS = auto(),
    LOSS = auto(),
    OTHER = auto()


@dataclass
class MarkerSequence:
    type: MarkerSequenceType
    near_fill_idxs: List[int] = None


class GameCategory(Enum):
    WIN = auto(),
    NEAR_WIN = auto(),
    NEAR_FORK_WIN = auto(),
    NEAR_LOSS = auto(),
    NEAR_FORK_LOSS = auto(),
    LOSS = auto(),
    ONE_MOVE_IN = auto(),
    OTHER = auto()


class Generator:
    def __init__(self):
        self.agent = MinimaxAgent()

    def _evaluate_marker_sequence(self,
                                  marker_sequence: List[int],
                                  next_player_marker: int,
                                  dim: Tuple[int, int] = (3, 3),
                                  num_markers_in_win: int = 3):
        # Organize marker locations
        marker_locs = defaultdict(list)
        for i, marker in enumerate(marker_sequence):
            marker_locs[marker].append(i)

        # Opp player
        if next_player_marker == GeneratorMarker.PLAYER1.value:
            opp_player = GeneratorMarker.PLAYER2.value
        else:
            opp_player = GeneratorMarker.PLAYER1.value

        # Win
        if len(marker_locs[next_player_marker]) == len(marker_sequence):
            return MarkerSequence(MarkerSequenceType.WIN)
        
        # Near win
        if (len(marker_locs[next_player_marker]) == len(marker_sequence) - 1
            and len(marker_locs[opp_player]) == 0):
            return MarkerSequence(MarkerSequenceType.NEAR_WIN, marker_locs[GeneratorMarker.EMPTY.value])
        
        # Near fork win
        if (len(marker_locs[next_player_marker]) == len(marker_sequence) - 2
            and len(marker_locs[opp_player]) == 0):
            return MarkerSequence(MarkerSequenceType.NEAR_FORK_WIN, marker_locs[GeneratorMarker.EMPTY.value])
        
        # Near fork loss
        if (len(marker_locs[opp_player]) == len(marker_sequence) - 2
            and len(marker_locs[next_player_marker]) == 0):
            return MarkerSequence(MarkerSequenceType.NEAR_FORK_LOSS, marker_locs[GeneratorMarker.EMPTY.value])
        
        # Near loss
        if (len(marker_locs[opp_player]) == len(marker_sequence) - 1
            and len(marker_locs[next_player_marker]) == 0):
            return MarkerSequence(MarkerSequenceType.NEAR_LOSS, marker_locs[GeneratorMarker.EMPTY.value])
        
        # Loss 
        if len(marker_locs[opp_player]) == len(marker_sequence): 
            return MarkerSequence(MarkerSequenceType.LOSS)
        
        return MarkerSequence(MarkerSequenceType.OTHER)
    
    def get_all_valid_moves(self, game_state): 
        # returns valid moves from 1, 2,...,9 
        clear_pos = []
        for row in range(len(game_state)):
            for col in range(len(game_state)):
                if game_state[row][col] == 0: 
                    clear_pos.append(row * 3 + col + 1)
        return clear_pos

    def get_random_sequence(self, game_state): 
        player_1_squares = []
        player_2_squares = []
        for i in range(len(game_state)):
            for j in range(len(game_state)): 
                if game_state[i][j] == 1: 
                    player_1_squares.append(i * 3 + j + 1)
                elif game_state[i][j] == -1: 
                    player_2_squares.append(i * 3 + j + 1)
        random_1 = random.shuffle(player_1_squares)
        random_2 = random.shuffle(player_2_squares)
        interleaved_sequence = []
        if random_1 is not None: 
            if random_2 is not None: 
                for p1, p2 in zip(random_1, random_2):
                    interleaved_sequence.extend([p1, p2])
                # Add the remaining element from player1_squares if any
                if len(random_1) > len(random_2):
                    interleaved_sequence.append(random_1[-1])
            if len(random_1) == 1: 
                interleaved_sequence.append(random_1[-1])
        return interleaved_sequence

    def _get_game_category(self,
                           game_state: List[List[int]],
                           next_player_marker: int,
                           dim: Tuple[int, int] = (3, 3),
                           num_markers_in_win: int = 3):
        assert len(dim) == 2
        width, height = dim

        near_loss_exists = False
        player_fork_locs = defaultdict(int)
        opp_fork_locs = defaultdict(int)

        # print(game_state)
        random_sequence = self.get_random_sequence(game_state)

        if len(random_sequence) == 1: 
            return GameCategory.ONE_MOVE_IN

        best_moves, utility = self.agent.get_best_next_moves(random_sequence)

        # Opp player
        if next_player_marker == GeneratorMarker.PLAYER1.value:
            cur_player = GeneratorMarker.PLAYER1.value
            opp_player = GeneratorMarker.PLAYER2.value
        else:
            opp_player = GeneratorMarker.PLAYER1.value
            cur_player = GeneratorMarker.PLAYER2.value

        if utility * cur_player > 0: 
            if len(best_moves) == 0: 
                return GameCategory.WIN
            # elif there is a near win 
            near_win = 0
            for row_idx in range(3): 
                row = game_state[row_idx]
                col_idx = row_idx
                column = [game_state[i][col_idx] for i in range(3)]
                if sum(column) == 2 * cur_player or sum(row) == 2 * cur_player: 
                    near_win = 1
            # check diagonals consecutive near-win 
            if (game_state[0][0] + game_state[1][1] + game_state[2][2] == 2 * cur_player or game_state[0][2] + game_state[1][1] + game_state[2][0] == 2 * cur_player): 
                near_win = 1
            if near_win == 1: 
                return GameCategory.NEAR_WIN
            # else there is a near fork win 
            near_fork_win = 0
            if game_state[0][0] == cur_player and game_state[2][2] == cur_player:
                if game_state[0][1] == 0 and game_state[1][2] == 0 and game_state[0][2] == 0: 
                    near_fork_win = 1
                if game_state[1][0] == 0 and game_state[2][1] == 0 and game_state[2][0] == 0: 
                    near_fork_win = 1
            if game_state[0][2] == cur_player and game_state[2][0] == cur_player: 
                if game_state[0][1] == 0 and game_state[1][0] == 0 and game_state[0][0] == 0: 
                    near_fork_win = 1
                if game_state[1][2] == 0 and game_state[2][1] == 0 and game_state[2][2] == 0: 
                    near_fork_win = 1
            if near_fork_win == 1: 
                return GameCategory.NEAR_FORK_WIN
            else:
                return GameCategory.OTHER

        elif utility * cur_player < 0: 
            if len(best_moves) == 0: 
                return GameCategory.LOSE
            # elif there is a near win 
            near_loss = 0
            for row_idx in range(3): 
                row = game_state[row_idx]
                col_idx = row_idx
                column = [game_state[i][col_idx] for i in range(3)]
                if sum(column) == 2 * opp_player or sum(row) == 2 * opp_player: 
                    near_loss = 1
            # check diagonals consecutive near-win 
            if (game_state[0][0] + game_state[1][1] + game_state[2][2] == 2 * cur_player or game_state[0][2] + game_state[1][1] + game_state[2][0] == 2 * opp_player): 
                near_loss = 1
            if near_loss == 1: 
                return GameCategory.NEAR_LOSS

        elif utility * cur_player == 0: 
            potential_near_fork_loss = 0
            # check for a near fork loss 
            if game_state[0][0] == opp_player and game_state[2][2] == opp_player:
                if game_state[0][1] == 0 and game_state[1][2] == 0 and game_state[0][2] == 0: 
                    potential_near_fork_loss = 1
                if game_state[1][0] == 0 and game_state[2][1] == 0 and game_state[2][0] == 0: 
                    potential_near_fork_loss = 1
            if game_state[0][2] == opp_player and game_state[2][0] == opp_player: 
                if game_state[0][1] == 0 and game_state[1][0] == 0 and game_state[0][0] == 0: 
                    potential_near_fork_loss = 1
                if game_state[1][2] == 0 and game_state[2][1] == 0 and game_state[2][2] == 0: 
                    potential_near_fork_loss = 1
            if potential_near_fork_loss == 1: 
                # pick a suboptimal move 
                all_valid_moves = self.get_all_valid_moves(game_state)
                bad_moves = [move for move in all_valid_moves if move not in best_moves]
                if bad_moves == []: 
                    return GameCategory.NEAR_FORK_LOSS
                # print(game_state, bad_moves)
                bad_move = random.choice(bad_moves)
                new_sequence = random_sequence.extend(bad_move)
                new_utility, new_best_moves = self.agent.get_best_next_moves(new_sequence)
                # play the suboptimal move and ask the minimax again 
                if new_utility * opp_player > 0: 
                    return GameCategory.NEAR_FORK_LOSS

        return GameCategory.OTHER

    def _generate_all_valid_states(self,
                                   game_state: List[List[int]],
                                   num_markers_on_state: int,
                                   game_state_accumulator: Set[Tuple[Tuple[int]]],
                                   game_state_categories : Dict[GameCategory, Tuple[Tuple[int]]],
                                   dim: Tuple[int, int] = (3, 3),
                                   num_markers_in_win: int = 3):
        # Prune: state already appears
        key = tuple(tuple(row) for row in game_state)
        if key in game_state_accumulator:
            return
        game_state_accumulator.add(key)

        # Determine player turn
        next_player = (num_markers_on_state % 2) + 1
        next_player_marker = [GeneratorMarker.EMPTY, 
                              GeneratorMarker.PLAYER1.value, 
                              GeneratorMarker.PLAYER2.value][next_player]

        # Action on game category
        game_category = self._get_game_category(game_state, next_player_marker, dim=dim, num_markers_in_win=num_markers_in_win)

        if game_category == GameCategory.WIN or game_category == GameCategory.LOSS:
            return
        elif num_markers_on_state == 1:
            game_state_categories[GameCategory.ONE_MOVE_IN].append(key)
        else:
            game_state_categories[game_category].append(key)
    
        # Backtrack
        for row in range(len(game_state)):
            for col in range(len(game_state[row])):
                if game_state[row][col] == GeneratorMarker.EMPTY.value:
                    # Choose
                    game_state[row][col] = next_player_marker

                    # Explore
                    self._generate_all_valid_states(game_state, 
                                                    num_markers_on_state + 1, 
                                                    game_state_accumulator,
                                                    game_state_categories,
                                                    dim=dim, 
                                                    num_markers_in_win=num_markers_in_win)
                    
                    # Unchoose
                    game_state[row][col] = GeneratorMarker.EMPTY.value


    def _generate_all_valid_states_wrapper(self,
                                           dim: Tuple[int, int] = (3, 3),
                                           num_markers_in_win: int = 3) -> Set[Tuple[Tuple[int]]]:
        assert len(dim) == 2
        width, height = dim

        game_state_accumulator = set()
        game_state_categories = defaultdict(list)
        initial_state = [[GeneratorMarker.EMPTY.value for _ in range(width)] for _ in range(height)]
        self._generate_all_valid_states(initial_state, 
                                        0,
                                        game_state_accumulator, 
                                        game_state_categories,
                                        dim=dim, 
                                        num_markers_in_win=num_markers_in_win)
        return game_state_categories


    def _state_to_any_sequence(self,
                               game_state: List[List[int]]) -> List[int]:
        pass

    
    def sample_category(self,
                        category: GameCategory):
        pass

    def sample_n_all_categories(self):
        pass

generator = Generator()
valid_games = generator._generate_all_valid_states_wrapper()
print(valid_games[GameCategory.ONE_MOVE_IN])
print(len(valid_games[GameCategory.WIN]))
print(len(valid_games[GameCategory.LOSS]))
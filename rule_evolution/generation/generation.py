from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def sample(self, num_examples_per_category : int):
        pass


"""
# Partition sequences into:
# near loss to fork
# near win by fork
# near loss to consecutive
# near win by consecutive
# starting positions (opponent makes suboptimal move in response to starting move / respond to opponent's first move)
# total set of sequences
import json
import copy
import csv
import random

import sys 
sys.path.append('../util')

from game_util import MinimaxAgent

MAX_DEPTH = 9
PLAYER_1_WINS = 1
PLAYER_2_WINS = -1
DRAW_STATE = 0
TERMINAL_STATES = {PLAYER_2_WINS, PLAYER_1_WINS, DRAW_STATE}
NON_TERMINAL_STATE = None
ALL_BOARD_POSITIONS = {1, 2, 3, 4, 5, 6, 7, 8, 9}

PLAYER_1_TOKEN = 1
PLAYER_2_TOKEN = -1


class Game:
    def __init__(self):
        # for eg. when player 1 plays to position 6, the sixth 0 from left will be replaced w 1.
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.sequence = []

        self.tags = {"player_1_near_win_consecutive" : 0, 
                "player_2_near_win_consecutive" : 0,
                "player_1_near_fork": 0,
                "player_2_near_fork": 0,
                "starting_pos" : 0}

        # set initial state to non-terminal
        self.state = NON_TERMINAL_STATE

        self.cur_player = 1

        # used to find if a generated state is a losing one. to exclude
        self.agent = MinimaxAgent()

    # def partition_games(self):
        # for game in self.generate_games():
            # self.tag_game()
    
    def tag_game(self):
        if len(self.sequence) == 1: 
            self.tags["starting_pos"] = 1
        else: 
            self.tags["starting_pos"] = 0

        # check for near wins consecutive 
        bit_1 = 0
        bit_2 = 0 
        for row_idx in range(3): 
            row = self.board[row_idx]
            if sum(row) == 2 * PLAYER_1_TOKEN: 
                bit_1 = 1
            if sum(row) == 2 * PLAYER_2_TOKEN: 
                bit_2 = 1
            col_idx = row_idx
            column = [self.board[i][col_idx] for i in range(3)]
            if sum(column) == 2 * PLAYER_1_TOKEN: 
                bit_1 = 1
            if sum(column) == 2 * PLAYER_2_TOKEN: 
                bit_2 = 1
                
        # check diagonals consecutive near-win 
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == 2: 
            bit_1 = 1
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == -2: 
            bit_2 = 1
        
        # check for forks
        bit_1_fork = 0
        bit_2_fork = 0
        if self.board[0][0] == 1 and self.board[2][2] == 1:
            if self.board[0][1] == 0 and self.board[1][2] == 0 and self.board[0][2] == 0: 
                bit_1_fork = 1
            if self.board[1][0] == 0 and self.board[2][1] == 0 and self.board[2][0] == 0: 
                bit_1_fork = 1
        if self.board[0][2] == 1 and self.board[2][0] == 1: 
            if self.board[0][1] == 0 and self.board[1][0] == 0 and self.board[0][0] == 0: 
                bit_1_fork = 1
            if self.board[1][2] == 0 and self.board[2][1] == 0 and self.board[2][2] == 0: 
                bit_1_fork = 1
        
        if self.board[0][0] == -1 and self.board[2][2] == -1:
            if self.board[0][1] == 0 and self.board[1][2] == 0 and self.board[0][2] == 0: 
                bit_2_fork = 1
            if self.board[1][0] == 0 and self.board[2][1] == 0 and self.board[2][0] == 0: 
                bit_2_fork = 1
        if self.board[0][2] == -1 and self.board[2][0] == -1: 
            if self.board[0][1] == 0 and self.board[1][0] == 0 and self.board[0][0] == 0: 
                bit_2_fork = 1
            if self.board[1][2] == 0 and self.board[2][1] == 0 and self.board[2][2] == 0: 
                bit_2_fork = 1
        # if the state is a near win, it should not be tagged as a near fork 
        near_win = 0
        if bit_1 == 1: 
            self.tags["player_1_near_win_consecutive"] = 1
            near_win = 1
        if bit_2 == 1: 
            self.tags["player_2_near_win_consecutive"] = 1
            near_win = 1
        if near_win == 0:
            if bit_1_fork == 1:
                self.tags["player_1_near_fork"] = 1
            if bit_2_fork == 1: 
                self.tags["player_2_near_fork"] = 1
        return self.tags

    def copy(self):
        return copy.deepcopy(self)

    def get_game_state(self):
        # Func returns the current game state
        for row in self.board:
            if row == [-1, -1, -1]:
                self.state = PLAYER_2_WINS
            if row == [1, 1, 1]:
                self.state = PLAYER_1_WINS

        for col in range(3):
            column = [row[col] for row in self.board]
            if column == [-1, -1, -1]:
                self.state = PLAYER_2_WINS
            if column == [1, 1, 1]:
                self.state = PLAYER_1_WINS
        # check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != 0:
            if self.board[0][0] == 1:
                self.state = PLAYER_1_WINS
            else:
                return PLAYER_2_WINS
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != 0:
            if self.board[0][0] == 1:
                self.state = PLAYER_1_WINS
            else:
                self.state = PLAYER_2_WINS

        if len(self.sequence) == 9:
            self.state = DRAW_STATE

        self.state = NON_TERMINAL_STATE
        return self.state

    def is_game_over(self):
        return self.get_game_state() in TERMINAL_STATES

    def get_all_possible_moves(self):
        clear_pos = []
        for pos in ALL_BOARD_POSITIONS:
            if pos not in self.sequence:
                clear_pos.append(pos)
        return clear_pos

    def get_best_moves(self): 
        best_moves, utility = self.agent.get_best_next_moves(self.game_sequence)
        return best_moves

    def make_move(self, move):
        self.sequence.append(move)
        row_index = (move - 1) // 3
        col_index = (move - 1) % 3
        self.board[row_index][col_index] = self.cur_player
        if self.cur_player == 1:
            self.cur_player = -1
        else:
            self.cur_player = 1

class Generator:
    def __init__(self):
        # dict of games, partitioned by game category, containing list of games. a game is a sequence of numbers
        self.games = {"player_1_near_win_consecutive" : {}, 
                "player_2_near_win_consecutive" : {},
                "player_1_near_fork": {},
                "player_2_near_fork": {},
                "starting_pos" : {}}
        
        #self.generate_all_valid_games()

    # samples random tag and a random game from that tag. returns a game sequence
    def sample_game(self): 
        with open('partitioned_all_2.json') as f:
            all_games = json.load(f)
        # random_key = random.choice(list(all_games.keys()))
        random_key = random.choice(list(all_games.keys()).extend('all'))
        if random_key == 'all': 
            # sample from the all_sequences_formatted
            with open('all_sequences_formatted_2.json') as f2: 
                all_games = json.load(f2)
                random_board = random.choice(all_games['all'])
                random_game_sequence = random.choice(all_games[random_key])
                return random_game_sequence
        random_board = random.choice(all_games[random_key])
        random_game_sequence = random.choice(all_games[random_key][random_board])
        return random_game_sequence
    
    # to get a sequence of the examples per specified tag 
    def sample_n_games(self, n): 
        games = []
        for _ in range(n): 
            games.append(self.sample_game())
        return games 

    def get_board_str(self, board): 
        # gets the string representation of board so that we can hash the string in the partitioned table of games
        board_str = ""
        for i in range(3): 
            for j in range(3): 
                if board[i][j] == 1: 
                    board_str += '1'
                if board[i][j] == -1: 
                    board_str += '2'
                if board[i][j] == 0: 
                    board_str += '0'
        return board_str

    def partition_games(self, games = None):
        print("Partitioning...")
        possible_tags = ["starting_pos", "player_1_near_win_consecutive", "player_2_near_win_consecutive", "player_1_near_fork", "player_2_near_fork"]
        if not games:
            with open('all_sequences_formatted_correctly_2.json') as f:
                games = json.load(f)
                
                for game in games["all"]:
                    # Convert the line from JSON to a Python dictionary
                    # game = json.loads(line.strip())
                    for tag in possible_tags:
                        if game["tags"][tag] == 1:
                            board_str = self.get_board_str(game["board"])
                            if board_str not in self.games[tag]: 
                                self.games[tag][board_str] = []
                            self.games[tag][board_str].append(game["sequence"]) # just grab the sequence
            print("Finished partitioning from file!")
        else:
            for game in games:
                for tag in possible_tags:
                    if game.tags[tag] == 1:
                        board_str = self.get_board_str(game["board"])
                        if board_str not in self.games[tag]: 
                            self.games[tag][board_str] = []
                        self.games[tag][board_str].append(game["sequence"]) # just grab the sequence
            print("Finished partitioning from dict!")

        print(self.games)
        with open('partitioned_all_2.json', 'w') as file:
            json.dump(self.games, file)
        print("Saved partition!")
    
    def generate_all_valid_games(self):
        initial_game = Game()
        print("Generating all valid games...")
        all_valid_games = []
        last_line = 8549
        counter = 0
        with open('all_sequences_formatted_2.json', 'w') as file:
            file.write('{\n\t"all": [\n\t\t')

            for out in self.generate_games(initial_game):
                all_valid_games.append(out)
                counter += 1
                # if counter < last_line:
                file.write(json.dumps(out) + ',\n\t\t')
                # else: 
                    # file.write(json.dumps(out) + '\n')
        file.write('\t]\n}')
        print("Finished generating all valid games!")
        
        self.partition_games(all_valid_games)
        
    def generate_games(self, game, game_sequence=None, depth=0):
        if game_sequence is None:
            game_sequence = []

        # skip losing games
        best_moves, utility = game.agent.get_best_next_moves(game_sequence)
        if [0, 1, -1][len(game_sequence) % 2 + 1] * utility < 0:
            return

        if game.is_game_over() or depth > MAX_DEPTH - 1:
            # don't yield, game is over or there's only one possible move left 
            return
        
        output = {
            "sequence": game.sequence, 
            "tags": game.tag_game(),
            "board": game.board
        }
        yield output

        # best_moves, _ = game.agent.get_best_next_moves(game_sequence)
        valid_moves = game.get_all_possible_moves()
        for move in valid_moves:
            new_game = game.copy()
            new_game.make_move(move)
            yield from self.generate_games(new_game, game_sequence + [move], depth + 1)

def write_games_to_files():
    generator = Generator()
    generator.generate_all_valid_games()

    with open('all_sequences.json', 'w') as file:
        for game_sequence in generator.generate_games(initial_game):
            file.write(json.dumps(game_sequence) + '\n')
    

    with open('all_sequences.txt', 'w') as file:
        # Write data from the generator to the text file
        for game_sequence in Generator.generate_games(initial_game):
            # Write each piece of data on a new line
            file.write(str(game_sequence) + '\n')

    end_results = []
    with open('all_sequences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for game_sequence in Generator.generate_games(initial_game):
            *sequence, result = game_sequence
            end_results.append(result)
            i = 9 - len(sequence)
            writer.writerow(sequence + [''] * i + [result])


if __name__ == '__main__':
    write_games_to_files()
    # generator = Generator()
    # generator.partition_games()
        

def check_near_wins_and_forks(grid, marker, n):
    m, n_cols = len(grid), len(grid[0])
    potential_forks = []

    # Helper function to check if a list has n-1 markers and 1 empty space
    def is_near_win(lst):
        return lst.count(marker) == n - 1 and lst.count(None) == 1

    # Add a potential fork position
    def add_fork_position(lst, positions):
        if is_near_win(lst):
            potential_forks.extend(positions)

    # Check rows and columns for near-wins
    for i in range(m):
        for j in range(n_cols - n + 1):
            # Check row segment
            row_segment = [grid[i][j + k] for k in range(n)]
            row_positions = [(i, j + k) for k in range(n) if grid[i][j + k] is None]
            add_fork_position(row_segment, row_positions)

    for j in range(n_cols):
        for i in range(m - n + 1):
            # Check column segment
            col_segment = [grid[i + k][j] for k in range(n)]
            col_positions = [(i + k, j) for k in range(n) if grid[i + k][j] is None]
            add_fork_position(col_segment, col_positions)

    # Check diagonals for near-wins
    for i in range(m - n + 1):
        for j in range(n_cols - n + 1):
            # Main diagonal segment
            main_diag_segment = [grid[i + k][j + k] for k in range(n)]
            main_diag_positions = [(i + k, j + k) for k in range(n) if grid[i + k][j + k] is None]
            add_fork_position(main_diag_segment, main_diag_positions)

            # Anti-diagonal segment
            anti_diag_segment = [grid[i + k][j + n - 1 - k] for k in range(n)]
            anti_diag_positions = [(i + k, j + n - 1 - k) for k in range(n) if grid[i + k][j + n - 1 - k] is None]
            add_fork_position(anti_diag_segment, anti_diag_positions)

    # Find fork positions
    fork_positions = set(potential_forks)
    forks = [pos for pos in fork_positions if potential_forks.count(pos) > 1]

    return forks
"""
import json
import copy
import csv
MAX_DEPTH = 9
WIN_STATE = 1
LOSE_STATE = -1
DRAW_STATE = 0
TERMINAL_STATES = {WIN_STATE, LOSE_STATE, DRAW_STATE}
NON_TERMINAL_STATE = None

ALL_BOARD_POSITIONS = {1, 2, 3, 4, 5, 6, 7, 8, 9}

# Win means that player 2 won, lose means that player 2 lost, etc.

# [1, 2, 3, ..., 9] -> player 1 plays to 1, player 2 plays to 2, player 1 plays to 3, so on until game termination
# would want to render it in 3x3 form as so: ?
# positions
# [ [1, 2, 3],
#   [4, 5, 6],
#   [7, 8, 9] ]

class Game:
    def __init__(self):
        # for eg. when player 1 plays to position 6, the sixth 0 from left will be replaced w 1.
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.sequence = []

        # set initial state to non-terminal
        self.state = NON_TERMINAL_STATE

        self.cur_player = 1

    def copy(self):
        return copy.deepcopy(self)

    def get_game_state(self):
        # Func returns the current game state
        for row in self.board:
            if row == [2, 2, 2]:
                return WIN_STATE
            if row == [1, 1, 1]:
                return LOSE_STATE

        for col in range(3):
            column = [row[col] for row in self.board]
            if column == [2, 2, 2]:
                return WIN_STATE
            if column == [1, 1, 1]:
                return LOSE_STATE
        # check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != 0:
            if self.board[0][0] == 1:
                return LOSE_STATE
            else:
                return WIN_STATE
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != 0:
            if self.board[0][0] == 1:
                return LOSE_STATE
            else:
                return WIN_STATE

        if len(self.sequence) == 9:
            return DRAW_STATE

        return NON_TERMINAL_STATE

    def is_game_over(self):
        if self.get_game_state() in TERMINAL_STATES:
            return True
        else:
            return False

    def get_all_possible_moves(self):
        clear_pos = []
        for pos in ALL_BOARD_POSITIONS:
            if pos not in self.sequence:
                clear_pos.append(pos)
        return clear_pos

    def make_move(self, move):
        self.sequence.append(move)
        row_index = (move - 1) // 3
        col_index = (move - 1) % 3
        self.board[row_index][col_index] = self.cur_player
        if self.cur_player == 1:
            self.cur_player = 2
        else:
            self.cur_player = 1

def generate_games(game, game_sequence=None, depth=0):
    if game_sequence is None:
        game_sequence = []

    if game.is_game_over() or depth > MAX_DEPTH:
        yield game_sequence + [game.get_game_state()]
        return

    for move in game.get_all_possible_moves():
        new_game = game.copy()
        new_game.make_move(move)
        yield from generate_games(new_game, game_sequence + [move], depth + 1)


def write_games_to_files():
    initial_game = Game()

    with open('all_full_sequences.json', 'w') as file:
        for game_sequence in generate_games(initial_game):
            file.write(json.dumps(game_sequence) + '\n')

    with open('all_full_sequences.txt', 'w') as file:
        # Write data from the generator to the text file
        for game_sequence in generate_games(initial_game):
            # Write each piece of data on a new line
            file.write(str(game_sequence) + '\n')

    end_results = []
    with open('all_full_sequences.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for game_sequence in generate_games(initial_game):
            *sequence, result = game_sequence
            end_results.append(result)
            i = 9 - len(sequence)
            writer.writerow(sequence + [''] * i + [result])
        # writer.writerow([''] * 9 + [', '.join(map(str, end_results))])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    write_games_to_files()

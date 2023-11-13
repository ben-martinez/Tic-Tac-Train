import random

from generation import Generator

PLAYER_1_WINS = 1
PLAYER_2_WINS = -1
DRAW_STATE = 0

N_GAMES = 3

class ManualGenerator(Generator):
	def __init__(self):
		# state is a list of list of ints...
		self.critical_states = {
		"starting-pos": [[[1, 0, 0], [0, 0, 0], [0, 0, 0]], 
				[[0, 0, 0], [0, 1, 0], [0, 0, 0]], 
				[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
		"near-win": [
			[[1, -1, -1], 
			[1, 1, 0], 
			[0, 0, -1]],

			[[1, 0, 0],
			[-1, 1, 0],
			[-1, 0, 0]],

			[[0, 1, -1],
			[1, 1, 0],
			[0, 0, -1]],

			[[1, -1, 0], 
			[1, 1, 0], 
			[-1, 0, -1]],

			[[1, 0, -1], 
			[0, 1, -1], 
			[1, 0, 0]],
		],
		"near-loss": [
			[[1,0,0],
			[0,-1,-1],
			[0,0,1]],

			[[1, -1, 1],
			[0, 1, 0],
			[0, 1, -1]],

			[[0, 1, -1],
			[1, 1, -1],
			[-1, 0, 1]],

			[[-1, -1, 1],
			[0, 1, 1],
			[0, 0, 0]],

			[[0, -1, 1],
			[-1, 1, 0],
			[0, 1, 0]]



		],
		"near-fork-win": [
			[[1, 0, -1],
			[0, -1, 0],
			[0, 0, 1]],
			[[1, 0, -1],
			[0, -1, 0],
			[0, 0, 1]],
			[[0, 1, 0],
			[1, -1, -1],
			[0, 0, 0]]
		],
		"near-fork-loss": [
			[[0, 0, 0],
			[-1, 0, 1],
			[1, 0, 0]],

			[[1,0,0],
			[0,-1,0],
			[0,0,1]],

			[[0,1,0],
			[0,0,0],
			[-1,0,1]],
		],
		"draw-vs-loss": [
			[[-1, 0, 1],
			[1, 1, -1],
			[0, -1, 1]], 

			[[1, 0, -1],
			[-1, -1, 1],
			[1, -1, 0]], 

			[[1, -1, 0],
			[-1, 1, 0],
			[1, 1, -1]], 
		]
		}

	# Symmetrical transformations
	def rotate_grid_90(self, grid):
		# Transpose the grid
		transposed_grid = [list(row) for row in zip(*grid)]
		# Reverse each row to get a 90 degree rotation
		rotated_grid = self.reflect_grid_vertical(transposed_grid)
		return rotated_grid

	def reflect_grid_horizontal(self, grid):
		# Reflect the grid horizontally by reversing the order of the rows
		reflected_grid = grid[::-1]
		return reflected_grid
	def reflect_grid_vertical(self, grid):
		# Reflect each row by reversing it
		reflected_grid = [row[::-1] for row in grid]
		return reflected_grid
	

	# returns list of a random game under each critical state
	# doesn't apply a random transform
	def get_random_game_all_vanilla(self):
		random_games = [random.choice(self.critical_states[key]) for key in self.critical_states.keys()]
		return random_games 

	# returns list of random transformed game under each critical state
	def get_random_game_all_states(self): 
		random_games = self.get_random_game_all_vanilla()
		random_transformed_games = [self.apply_random_transform(game) for game in random_games]
		return random_transformed_games
        
	# gets one random game and applies a random transformation to it
	def get_random_game(self, state): 
		state_games = self.critical_states[state]
		game = random.choice(state_games)
		transformed_game = self.apply_random_transform(game)
		return transformed_game
	
	# Each call to random_transformation will apply a random set of transformations to the grid.
	def apply_random_transform(self, grid):
		# Randomly choose the number of 90 degree rotations (0-3 times)
		num_rotations = random.randint(0, 3)		
		apply_reflectx = random.choice([True, False])
		apply_reflecty = random.choice([True, False])

		for _ in range(num_rotations):
			grid = self.rotate_grid_90(grid)

		# Apply reflectx if chosen
		if apply_reflectx:
			grid = self.reflect_grid_horizontal(grid)

		# Apply reflecty if chosen
		if apply_reflecty:
			grid = self.reflect_grid_vertical(grid)

		return grid
	
	# utility function. returns a random sequence given a game board
	
	
	def get_random_sequence(self, game_state):
		player_1_squares = []
		player_2_squares = []
		for i in range(len(game_state)):
			for j in range(len(game_state[i])):
				if game_state[i][j] == 1: 
					player_1_squares.append(i * 3 + j + 1)
				elif game_state[i][j] == -1: 
					player_2_squares.append(i * 3 + j + 1)

		# Shuffle the moves in place
		random.shuffle(player_1_squares)
		random.shuffle(player_2_squares)

		# Interleave the moves
		interleaved_sequence = []
		for p1, p2 in zip(player_1_squares, player_2_squares):
			interleaved_sequence.extend([p1, p2])

		# Add the remaining moves if any
		interleaved_sequence.extend(player_1_squares[len(player_2_squares):])
		interleaved_sequence.extend(player_2_squares[len(player_1_squares):])

		return interleaved_sequence
	

	def sample(self, num_examples_per_category : int):
		eval_sequences = []
		
		for _ in range(num_examples_per_category):
			games = self.get_random_game_all_states()
			for game in games:
				eval_sequences.append(self.get_random_sequence(game))

		return eval_sequences

        
def generate_games(N):
	# N games per state. (6 states so Nx6 total games)
	manny_gen = ManualGenerator()
	games = []
	for _ in range(N):
		games += manny_gen.get_random_game_all_states()
	
	sequences = [manny_gen.get_random_sequence(game) for game in games]
	print(sequences)
	with open("manual_test_sequences.txt", 'w') as file:
        	for i in range(len(sequences)):
           		file.write(str(sequences[i]) + '\n')
		

def main():
	generate_games(N_GAMES)	
	print("Finished generating! wrote to file")


if __name__ == '__main__':
	main()
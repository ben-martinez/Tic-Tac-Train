from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

import asyncio
from tqdm import tqdm
import pickle
import os
import logging
import heapq
import random

import sys
sys.path.append('../evaluation')
sys.path.append('../util')
sys.path.append('../generation')

import game_util
import util
from evaluation import Evaluator, EvalResultBatch, EvalResultType
from gpt_util import LLMClient, LLMType
from generation import Generator
from manual_generation import ManualGenerator

api_key = None

# --- LOGGING ---
logger = util.get_basic_logger('evolution.py')
logger.setLevel(logging.DEBUG)

# --- EVOLVER ---
@dataclass
class Evolution:
    library: str
    eval_result_batch: EvalResultBatch
    formatting_score_cutoff: int = 0.7

    @property
    def score(self):
        # default score
        # return self.formatting_accuracy * self.best_move_accuracy**2
        if self.eval_result_batch.formatting_accuracy < self.formatting_score_cutoff:
            return 0.0
        
        return self.eval_result_batch.best_move_accuracy


    def __lt__(self, other):
        return self.score < other.score
    
# --- EVOLUTION SELECTION ---
def sample_top_k_curried(k):
    def sample_top_k(evolutions, evolution_heap):
        sorted_evolutions = sorted(evolutions, reverse=True)
        top_k = sorted_evolutions[:k]
        sampled = random.choice(top_k)
        return sampled

    return sample_top_k


def sample_top_k_across_generation_curried(k):
    def sample_top_k_across_generation(evolutions, evolution_heap):
        sorted_evolutions = sorted(evolution_heap, reverse=True)
        top_k = sorted_evolutions[:k]
        sampled = random.choice(top_k)
        return sampled

    return sample_top_k_across_generation


class EvolutionSelection(Enum):
    BEST_IN_GENERATION = [lambda evolutions, _: max(evolutions)]
    BEST_ACROSS_GENERATION = [lambda _, evolution_heap: max(evolution_heap)]
    TOP_3_SAMPLE = [sample_top_k_curried(3)]
    TOP_3_SAMPLE_ACROSS_GENERATION = [sample_top_k_across_generation_curried(3)]


class Evolver:
    RULE_LIBRARY_TAG = '<rule-library>'
    LEARN_TAG = '<learn>'

    GAME_STATE_TAG = '<game-state>'
    SUBOPTIMAL_MOVE_TAG = '<suboptimal-move>'
    OPTIMAL_MOVES_TAG = '<optimal-moves>'

    def __init__(self,
                 client: LLMClient,
                 generator: Generator,
                 evaluator: Evaluator,
                 *,
                 evolve_intro_prompt_path: str,
                 evolve_prompt_path: str,
                 failed_sequences_intro_prompt_path: str,
                 failed_sequence_prompt_path: str,
                 start_library_path: str = None):
        self.client = client
        self.generator = generator
        self.evaluator = evaluator

        with open(evolve_intro_prompt_path, 'r') as evolve_intro_prompt_file:
            self.evolve_intro_prompt = evolve_intro_prompt_file.read()

        with open(evolve_prompt_path, 'r') as evolve_prompt_file:
            self.evolve_prompt = evolve_prompt_file.read()
        assert self.evolve_prompt.find(self.RULE_LIBRARY_TAG) != -1
        assert self.evolve_prompt.find(self.LEARN_TAG) != -1

        with open(failed_sequences_intro_prompt_path, 'r') as failed_sequences_intro_prompt_file:
            self.failed_sequences_intro_prompt = failed_sequences_intro_prompt_file.read()

        with open(failed_sequence_prompt_path, 'r') as failed_sequence_prompt_file:
            self.failed_sequence_prompt = failed_sequence_prompt_file.read()
        assert self.failed_sequence_prompt.find(self.GAME_STATE_TAG) != -1
        assert self.failed_sequence_prompt.find(self.SUBOPTIMAL_MOVE_TAG) != -1
        assert self.failed_sequence_prompt.find(self.OPTIMAL_MOVES_TAG) != -1

        self.start_library = None
        if start_library_path != None:
            if os.path.exists(start_library_path):
                with open(start_library_path, 'r') as start_library_file:
                    self.start_library = start_library_file.read()

    
    def save_checkpoints(self,
                         population: int,
                         max_generations: int,
                         num_examples_per_category: int,
                         evolution_selection: EvolutionSelection,
                         model: str,
                         generation_idx: int, 
                         evolutions: List[Evolution], 
                         evolution_heap: List[Evolution],
                         config_tag: str = ''):
        # Automatically generate checkpoint directory name
        checkpoints_dir = f"checkpoints/{population}x{max_generations}x{num_examples_per_category}"

        evolution_selection_tag = ''.join([word[0].upper() for word in evolution_selection.name.split('.')[-1].split('_')])
        checkpoints_dir += f"_{evolution_selection_tag}"

        checkpoints_dir += f"_{model}"
        if config_tag:
            checkpoints_dir += f"_{config_tag}"

        # Save evolutions to directory
        for tag, evolution_group in [['gen', evolutions], ['best', evolution_heap]]:
            evolution_group = sorted(evolution_group, reverse=True)

            base_dir = f"{checkpoints_dir}/{tag}"
            file_name = f"checkpoint_{generation_idx}_{tag}"

            txt_dir = f"{base_dir}/txt"
            os.makedirs(txt_dir, exist_ok=True)
            with open(f"{txt_dir}/{file_name}.txt", 'w') as txt_file:
                for evolution in sorted(evolution_group, reverse=True):
                    txt_file.writelines(f"{evolution}\n\n")

            pkl_dir = f"{base_dir}/pkl"
            os.makedirs(pkl_dir, exist_ok=True)  
            with open(f"{pkl_dir}/{file_name}.pkl", 'wb') as pkl_file:
                pickle.dump(evolution_group, pkl_file)

    
    async def evolve_once(self, 
                          seed_evolution: Evolution,
                          eval_dataset: List[List[int]],
                          model: str) -> Evolution:
        # Create learn sequences
        failed_examples_message = self.failed_sequences_intro_prompt
        for eval_result in seed_evolution.eval_result_batch.eval_results:
            if eval_result.type == EvalResultType.SUBOPTIMAL_MOVE:
                game_state, _ = game_util.game_sequence_to_latex_state(eval_result.game_sequence, markers=[' ', 'X', 'O']) # TODO: marker retrieval
                suboptimal_move = game_util.sequence_number_to_location(eval_result.gpt_move)
                optimal_locs = [game_util.sequence_number_to_location(move) for move in eval_result.best_moves]

                failed_examples_message += util.fill_tags(self.failed_sequence_prompt,
                                                        {self.GAME_STATE_TAG : str(game_state),
                                                         self.SUBOPTIMAL_MOVE_TAG : str(suboptimal_move),
                                                        self.OPTIMAL_MOVES_TAG : str(optimal_locs)})

        # Evolve
        intro = self.evolve_intro_prompt
        example = util.fill_tags(self.evolve_prompt, {self.RULE_LIBRARY_TAG : seed_evolution.library,
                                                      self.LEARN_TAG : failed_examples_message})
        print(example)
        #evolved_library = await self.client.get_one_shot_completion('user', intro, example, model)
        evolved_library = "No rule library."

        # Accuracy
        eval_result_batch = await self.evaluator.evaluate_accuracy(evolved_library,
                                                                    [' ', 'X', 'O'], # TODO: marker retrieval
                                                                    eval_dataset,
                                                                    model)

        return Evolution(evolved_library, eval_result_batch)


    async def evolve(self,
                     population: int,
                     max_generations: int,
                     evolution_selection: EvolutionSelection,
                     model: str = None,
                     num_examples_per_category: int = 3,
                     num_best: int = 10,
                     show_progress: bool = True,
                     config_tag: str = ''):
        # Seed library
        seed_evolution = Evolution('The rule set does not yet exist.', EvalResultBatch(0.0, 0.0, []))
        if self.start_library != None:
            seed_evolution.library = self.start_library

        # Maintain top K evolutions across all generations
        evolution_heap = []

        # Evolve in parallel
        for generation_i in tqdm(range(max_generations)) if show_progress else range(max_generations):
            # Get eval dataset
            eval_dataset = self.generator.sample(num_examples_per_category)
            # eval_dataset = [[3, 5, 7]] * 1 # TODO remove this

            # Get evolutions
            evolution_requests = []
            for _ in range(population):
                request = self.evolve_once(seed_evolution, eval_dataset, model) # TODO: use eval dataset
                evolution_requests.append(request) # TODO: discard evolution with Nones
            evolutions : List[Evolution] = await asyncio.gather(*evolution_requests)

            # Update evolution heap
            for evolution in evolutions:
                heapq.heappush(evolution_heap, evolution)

            while len(evolution_heap) > num_best:
                heapq.heappop(evolution_heap)

            # Save checkpoint
            self.save_checkpoints(population, max_generations, num_examples_per_category, evolution_selection,
                                  model, generation_i, evolutions, evolution_heap, config_tag)

            # Select next seed
            seed_evolution: Evolution = evolution_selection.value[0](evolutions, evolution_heap)

            # Log evolutions
            if show_progress:
                logger.info(f"--- SEED EVOLUTION GENERATION {generation_i} LIBRARY: {seed_evolution.library}")
                logger.info(f"--- SEED EVOLUTION GENERATION {generation_i} FORMATTING ACCURACY: {seed_evolution.eval_result_batch.formatting_accuracy}")
                logger.info(f"--- SEED EVOLUTION GENERATION {generation_i} BEST MOVE ACCURACY: {seed_evolution.eval_result_batch.best_move_accuracy}")



with LLMClient(llm_type=LLMType.OPENAI, api_secret_key=api_key) as client:
    evaluator = Evaluator(client, 
                        '../evaluation/prompts/evaluation_rule_library.txt', 
                        '../evaluation/prompts/evaluation_example.txt')
    
    generator = ManualGenerator()
    
    # evolver = Evolver(client, 
    #                   generator, 
    #                   evaluator,
    #                   evolve_intro_prompt_path='prompts/default_evolution_intro_prompt.txt',
    #                   evolve_prompt_path='prompts/default_evolution_prompt.txt',
    #                   failed_sequences_intro_prompt_path='prompts/default_failed_sequences_intro_prompt.txt',
    #                   failed_sequence_prompt_path='prompts/default_failed_sequence_prompt.txt',
    #                   start_library_path='libraries/default_start_library.txt') # CHANGE START LIB
    # asyncio.run(evolver.evolve(5, 
    #                            1,
    #                            EvolutionSelection.BEST_ACROSS_GENERATION, 
    #                            model='gpt-3.5-turbo', 
    #                            num_examples_per_category=5,
    #                            config_tag='baseline'))
    num_examples_per_category = 5
    model = "gpt-4"
    eval_dataset = generator.sample(num_examples_per_category)

    rule_library = "No rule library"
    rule_library = "The following rule set includes some added and improved rules based on the given mistakes: \n\nRule 1 (Unchanged): \nGame Board Representation\n\nRule 2 (Unchanged): \nPlayer Moves \n\nRule 3 (Unchanged): \nWinning Conditions\n\nRule 4 (Improved):\nIn near-win states, the player should consider the potential threats of the opponent first before creating its own. \n\nRule 5 (Improved):\n- Among all available actions, play one of the moves that can lead to a win in the next turn. \n- If there is no winning move, block the opponent\'s move that can lead to a win in the next turn.\n- If the opponent doesn\'t have a winning move in the next turn, play a move that creates a two-in-a-row that cannot be immediately blocked by the opponent.\n- If there is no such move, block the opponent\'s move that can lead to a two-in-a-row that cannot be immediately blocked.\n- Potentially create a fork.\n\nRule 6 (Improved):\nA situation where you can win in two ways is called a \"fork.\" If there is no immediate threat or winning move, consider creating a fork.\n\nRule 7 (Improved):\nWhen there is no immediate threat or winning move and the opponent can create a fork, block it.\n\nRule 8 (Improved):\nIf no immediate threats or potential winning moves are possible, and the center square (if available) is free, take it.\n\nRule 9 (Improved): \nIf the center square is taken by the opponent and the opponent could win in the next move, consider making a move in a corner square that also blocks the opponent.\n\nRule 10 (New): \nWhile there is no immediate threat and no potential winning move, make a move that will block the opponent\'s favor. For example, if the opponent has occupied two corner squares, consider taking the side square of an empty row or column. If the opponent has occupied two side squares on a clear row or column, consider taking the corner square."

    valid_output_accuracy, best_move_accuracy = 0, 0

    num_iters = 5
    print("running: 5x5_MG_BIG_gpt-4.txt")
    for i in tqdm(range(num_iters)):
        eval_result = asyncio.run(evaluator.evaluate_accuracy(rule_library,
                                                                    [' ', 'X', 'O'], # TODO: marker retrieval
                                                                    eval_dataset,
                                                                    model))
        valid_output_accuracy_i, best_move_accuracy_i = eval_result.formatting_accuracy, eval_result.best_move_accuracy
        print(f"Iteration {i}'s output accuracy: {valid_output_accuracy_i},  best move accuracy: {best_move_accuracy_i}")
        valid_output_accuracy += valid_output_accuracy_i
        best_move_accuracy += best_move_accuracy_i
    valid_output_accuracy /= num_iters
    best_move_accuracy /= num_iters
    result = f"FINAL output accuracy: {valid_output_accuracy}, best move accuracy: {best_move_accuracy}\n"
    print(result)
    prev_res = "Previously it was formatting_accuracy=0.8333333333333334, best_move_accuracy=0.8666666666666667"

    with open(f"5x5_MG_BIG_gpt-4.txt", 'w') as txt_file:
        txt_file.write(result)
        txt_file.write(prev_res)
    


"""
    def get_top_k(self, accuracies, k): 
        # implements the top k filtering algorithm for this generation 
        # sort the overall accuracies from highest to lowest and tag them to their index in the generation 
        sorted_accuracies_with_indices = sorted(enumerate(accuracies), key=lambda x: x[1][0] * x[1][1], reverse=True)
        # Extract the sorted indices
        sorted_indices = [index for index, _ in sorted_accuracies_with_indices]
        return sorted_indices[:k]

    def epsilon_greedy(self, accuracies, epsilon): 
        # implements epsilon greedy algorithm for this generation 
        # epsilon = probability of choosing an item at random
        best_idx = max(range(len(accuracies)), key=lambda acc_idx: accuracies[acc_idx][0] * accuracies[acc_idx][1])
        exploration_bit = np.random.binomial(1, epsilon)
        if exploration_bit == 0:
            return best_idx 
        else: 
            idx_list = [i + 1 for i in range(population)]
            idx_list.remove(best_idx)
            random_int = random.randrange(population - 1)
            return idx_list[random_int]
"""

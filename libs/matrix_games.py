import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

def create_pure_strategy(n: int, a: int):
    assert n > 0 and a >= 0 and a < n 
    
    strat = np.zeros(n)
    strat[a] = 1
    return strat

def create_uniform_strategy(n: int):
    return np.array([1/n for _ in range(n)])

def create_prob_table(strat1: np.array, strat2: np.array):
    return np.outer(strat1,strat2)
    
def rps_matrix():
    return np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

def game_of_chicken():
    return np.array([[0,-1], [1,-10]]), np.array([[0,1],[-1,-10]])

def best_response_to_row_player(matrix: np.array, row_strategy: np.array):
    # Ensure that row_strategy is a 1D array
    row_strategy = np.array(row_strategy).reshape(1, -1)
    
    # Perform matrix multiplication
    payoffs = row_strategy @ matrix

    a = np.argmax(payoffs)
    return create_pure_strategy(matrix.shape[1], a)

def best_response_to_column_player(matrix: np.array, col_strategy: np.array):
    # Ensure that row_strategy is a 1D array
    col_strategy = np.array(col_strategy).reshape(-1, 1)
    
    # Perform matrix multiplication
    payoffs = matrix @ col_strategy

    a = np.argmax(payoffs)
    return create_pure_strategy(matrix.shape[0], a)

def compute_game_value(matrix1: np.array, matrix2: np.array, row: np.array, col: np.array):
    prob = create_prob_table(row, col)
    
    u1 = np.sum(matrix1 * prob)
    u2 = np.sum(matrix2 * prob)
    
    return u1, u2

def compute_zerosum_game_value(matrix: np.array, row: np.array, col: np.array):
    return compute_game_value(matrix, -matrix, row, col)

def best_response_value_function(matrix: np.array, step_size: float):
    import matplotlib.pyplot as plt

    steps = np.arange(0, 1 + step_size, step_size)
    vals = []
    for step in steps:
        new_strat = np.array([step, 1-step])
        br = best_response_to_row_player(-matrix, new_strat)

        p1_val, p2_val = compute_zerosum_game_value(matrix, new_strat, br)
        vals.append(p1_val)

    plt.scatter(steps, vals, s=10)
    plt.show()

def average_strat(strats: list):
    avg_strat = np.zeros(len(strats[0]))
    for strat in strats:
        avg_strat += strat
    
    avg_strat /= sum(avg_strat)
    return avg_strat


def best_response_to_average_row_strat(matrix: np.array, past_row_strategies: list):
    avg_strat = average_strat(past_row_strategies)
    return best_response_to_row_player(matrix, avg_strat)

def best_response_to_average_col_strat(matrix: np.array, past_col_strategies: list):
    avg_strat = average_strat(past_col_strategies)
    return best_response_to_column_player(matrix, avg_strat)

def best_reponse_to_last_row_strat(matrix: np.array, past_row_strategies: list):
    return best_response_to_row_player(matrix, past_row_strategies[-1])

def best_reponse_to_last_col_strat(matrix: np.array, past_col_strategies: list):
    return best_response_to_column_player(matrix, past_col_strategies[-1])


def self_play(matrix1: np.array, matrix2: np.array, iterations: int,
            strat_generator_for_row: Callable[[np.array, List[np.array]], np.array],
            strat_generator_for_col: Callable[[np.array, List[np.array]], np.array]) -> Tuple[list, list]:
    row_strategies, col_strategies, exploitabilities = [], [], []
    
    # start with uniform strategies
    row_strategies.append(create_uniform_strategy(matrix1.shape[0]))
    col_strategies.append(create_uniform_strategy(matrix2.shape[1]))
    
    for i in range(iterations):
        # create new strategies
        new_row_strat = strat_generator_for_row(matrix1, col_strategies)
        new_col_strat = strat_generator_for_col(matrix2, row_strategies)
                
        row_strategies.append(new_row_strat)
        col_strategies.append(new_col_strat)
        
        # calculate exploitability
        row = average_strat(row_strategies)
        col = average_strat(col_strategies)
        exploitabilities.append(calculate_exploitability(matrix1, matrix2, row, col))

    
    return row_strategies, col_strategies, exploitabilities


def calculate_deltas(matrix1: np.array, matrix2: np.array, row: np.array, col: np.array) -> np.array:
    u_row, u_col = compute_game_value(matrix1, matrix2, row, col)
    
    br_to_col = best_response_to_column_player(matrix1, col)
    br_to_row = best_response_to_row_player(matrix2, row)
    
    u_br_row, _ = compute_game_value(matrix1, matrix2, br_to_col, col)
    _, u_br_col = compute_game_value(matrix1, matrix2, row, br_to_row)
    
    return np.array([u_br_row - u_row, u_br_col - u_col])

def calculate_nash_conv(matrix1: np.array, matrix2: np.array, row: np.array, col: np.array) -> float:
    return np.sum(calculate_deltas(matrix1, matrix2, row, col))

def calculate_exploitability(matrix1: np.array, matrix2: np.array, row: np.array, col: np.array) -> float:
    return calculate_nash_conv(matrix1, matrix2, row, col)/2

def plot_exploitability(exploitabilities: np.array):
    plt.plot(list(range(len(exploitabilities))), exploitabilities)
    plt.show()
    
def plot_two_exploitability(expl1: np.array, expl2: np.array):
    plt.plot(list(range(len(expl1))), expl1)
    plt.plot(list(range(len(expl2))), expl2)
    plt.show()


def regret_minimization(matrix1: np.array, matrix2: np.array, iterations: int, regret_matching_alg: Callable[[np.array], np.array]):
    # prepare arrays to store regrets
    regrets_row = np.zeros(matrix1.shape[0])
    regrets_col = np.zeros(matrix2.shape[1])
    
    row_strategies, row_reward_vectors, exploitabilities = [], [], []
    col_strategies, col_reward_vectors = [], []
    row_regrets_storing, col_regrets_storing = [], []
    
    cumulative_reward_row, cumulative_reward_col = 0, 0

    for i in range(iterations):
        # prepare new strategies 
        new_row_strat = regret_matching_alg(regrets_row)
        row_strategies.append(new_row_strat)
        new_col_strat = regret_matching_alg(regrets_col)
        col_strategies.append(new_col_strat)
        
        # recieve reward vector
        reward_row = reward_vector_row(matrix1, new_col_strat)
        row_reward_vectors.append(reward_row)
        reward_col = reward_vector_col(matrix2, new_row_strat)
        col_reward_vectors.append(reward_col)
        
        
        # update cumulative reward
        current_reward_row = np.sum(new_row_strat * reward_row)
        cumulative_reward_row += current_reward_row
        current_reward_col = np.sum(new_col_strat * reward_col)
        cumulative_reward_col += current_reward_col
        
        # update regrets
        # update_regrets(regrets_row, reward_row, cumulative_reward_row)
        # update_regrets(regrets_col, reward_col, cumulative_reward_col)
        update_regrets(regrets_row, reward_row, current_reward_row)
        update_regrets(regrets_col, reward_col, current_reward_col)
        
        # store current regrets
        row_regrets_storing.append(reward_row - cumulative_reward_row)
        col_regrets_storing.append(reward_col - cumulative_reward_col)
        
        # calculate exploitability
        avg_row = average_strat(row_strategies)
        avg_col = average_strat(col_strategies)
        expl = calculate_exploitability(matrix1, matrix2, avg_row, avg_col)
        exploitabilities.append(expl)
        
    return row_strategies, col_strategies, exploitabilities, regrets_row, regrets_col, row_regrets_storing, col_regrets_storing
        

def reward_vector_row(matrix: np.array, col_strategy: np.array) -> np.array:
    return np.inner(matrix, col_strategy)

def reward_vector_col(matrix: np.array, row_strat: np.array) -> np.array:
    return np.inner(row_strat, matrix)

def regret_matching(regrets: np.array):
    positive_regrets = np.maximum(regrets, 0)
    total_positive = np.sum(positive_regrets)
    
    if total_positive == 0:
        n = len(regrets)
        return np.array([1/n for _ in range(n)])
    
    return positive_regrets / total_positive

def update_regrets(regrets: np.array, rewards: np.array, current_strat_reward: float):
    regrets += rewards - current_strat_reward
    

# def best_response_to_last_strat(past_opponent_strategies: list):
    
# matrix = rps_matrix()

# strats = [create_pure_strategy(3,0), create_pure_strategy(3,0), create_pure_strategy(3,1), create_pure_strategy(3,2)]
# print(strats)

# avg_strat = average_strat(strats)
# print(avg_strat)

# br_to_avg = best_response_to_average_row_strat(-matrix, strats)
# print(br_to_avg)

# br_to_avg = best_response_to_average_col_strat(matrix, strats)
# print(br_to_avg)


# m1 = rps_matrix()
# m2 = -m1
m1, m2 = game_of_chicken()
# m1 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
# m2 = -rps_matrix()

# row_strategy = np.array([0.1, 0.2, 0.7])
column_strategy = np.array([0.3, 0.2, 0.5])
row_strategy = np.array([0.3, 0.5, 0.2])

# delta_row, delta_column = calculate_deltas(matrix, -matrix, row_strategy, column_strategy)

# print(delta_row, delta_column)

# _,_,expl = self_play(matrix,-matrix,100,best_reponse_to_last_row_strat, best_reponse_to_last_col_strat)
# _,_,expl2 = self_play(matrix,-matrix,100,best_response_to_average_row_strat, best_reponse_to_last_col_strat)
# plot_two_exploitability(expl, expl2)
# print(row_strategy)
# print(matrix)

# reward = reward_vector_row(matrix, column_strategy)
# print(reward)
# value = np.sum(row_strategy * reward)
# print(value)

# regrets = np.zeros(len(row_strategy))
# print(f'before {regrets}')
# update_regrets(regrets, reward, value)
# print(f'after {regrets}')
# reward = reward_vector_col(matrix, row_strategy)
# print(reward)

rows, cols, expl, cum_regret_row, cum_regret_col, reg_row, reg_col = regret_minimization(m1, m2, 50, regret_matching)
plot_exploitability(expl)
# print(average_strat(rows))
print(rows)
print(expl[-1])

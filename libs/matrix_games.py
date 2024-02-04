import numpy as np
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from itertools import combinations
from ortools.linear_solver import pywraplp

# Strategy where action a is played with probability 1
def create_pure_strategy(n: int, a: int):
    assert n > 0 and a >= 0 and a < n 
    
    strat = np.zeros(n)
    strat[a] = 1
    return strat

# Strategy where each action has probability 1/(number of actions)
def create_uniform_strategy(n: int):
    return np.array([1/n for _ in range(n)])

# Probability distribution over possible pairs of actions
def create_prob_table(strat1: np.array, strat2: np.array):
    return np.outer(strat1,strat2)
    
# Rock-Paper-Scisors matrix for player 1
def rps_matrix():
    return np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

# Game of chicken, two matrices
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
    # Find probability distribution 
    prob = create_prob_table(row, col)
    
    # Calculate utilities with the given probability distribution
    u1 = np.sum(matrix1 * prob)
    u2 = np.sum(matrix2 * prob)
    
    return u1, u2

def compute_zerosum_game_value(matrix: np.array, row: np.array, col: np.array):
    return compute_game_value(matrix, -matrix, row, col)

def best_response_value_function(matrix: np.array, step_size: float):
    import matplotlib.pyplot as plt

    # matrix 2xN
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
    # Average strategy as sum of probabilities
    avg_strat = np.zeros(len(strats[0]))
    for strat in strats:
        avg_strat += strat
    
    # And normalized
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
    # Lists of strategies and exploitabilities
    row_strategies, col_strategies, exploitabilities = [], [], []
    
    # Start with uniform strategies
    row_strategies.append(create_uniform_strategy(matrix1.shape[0]))
    col_strategies.append(create_uniform_strategy(matrix2.shape[1]))
    
    for i in range(iterations):
        # Create new strategies
        new_row_strat = strat_generator_for_row(matrix1, col_strategies)
        new_col_strat = strat_generator_for_col(matrix2, row_strategies)
                
        row_strategies.append(new_row_strat)
        col_strategies.append(new_col_strat)
        
        # Calculate exploitability
        row = average_strat(row_strategies)
        col = average_strat(col_strategies)
        exploitabilities.append(calculate_exploitability(matrix1, matrix2, row, col))

    
    return row_strategies, col_strategies, exploitabilities


def calculate_deltas(matrix1: np.array, matrix2: np.array, row: np.array, col: np.array) -> np.array:
    # delta_i = u_i(br to -i, -i) - u_i(pi)
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
    # Prepare arrays to store regrets
    regrets_row = np.zeros(matrix1.shape[0])
    regrets_col = np.zeros(matrix2.shape[1])
    
    row_strategies, row_reward_vectors = [], []
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
        update_regrets(regrets_row, reward_row, current_reward_row)
        update_regrets(regrets_col, reward_col, current_reward_col)
        
        # store current regrets
        row_regrets_storing.append(reward_row - cumulative_reward_row)
        col_regrets_storing.append(reward_col - cumulative_reward_col)
        
        # calculate exploitability
        # avg_row = average_strat(row_strategies)
        # avg_col = average_strat(col_strategies)
        # expl = calculate_exploitability(matrix1, matrix2, avg_row, avg_col)
        # exploitabilities.append(expl)
        
    return row_strategies, col_strategies, regrets_row, regrets_col, row_regrets_storing, col_regrets_storing
        

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
    

def verify_support(matrix: np.array, support_row: np.array, support_col: np.array, for_row=True):
    submatrix = matrix.T[support_row][:, support_col]
    # print(submatrix)
    # print()
    if for_row:
        result = verify_matrix(submatrix.T)
    else:
        result = verify_matrix(submatrix)
    
    if result.success:
        return result.x[1:]
    return None
    
def verify_matrix(matrix: np.array):
    num_rows, num_cols = matrix.shape
    
    # 1*utility + matrix
    # A_eq = np.hstack([np.array([[1] * num_cols]).T, matrix])
    # A_eq = np.vstack([A_eq, [0] + [1] * num_rows])
    A_eq = np.hstack([np.array([[1] * num_rows]).T, matrix])
    A_eq = np.vstack([A_eq, [0] + [1] * num_cols])
    print(A_eq)
    print()
    b_eq = [0] * num_rows + [1]
    # print(b_eq)
    # print()
    c = np.zeros(shape=(num_cols + 1))
    c[0] = 1
    # print(c)
    # print()
    
    bounds = [(None, None)] + [(0, None) for _ in range(num_cols)]
    # print(bounds)
    # print()
    
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # print(res)
    return res
    
    
    
    
    
    
    
    
    
    
    # A_eq = np.ones(shape=(1, num_rows + 1))
    # A_eq[0, -1] = 0
    
    # b_eq = 1
    
    
    # A_ub = np.hstack(
    #     (-matrix.T, np.ones(shape=(num_cols, 1)))
    # )
    # b_ub = np.zeros(shape=(num_cols, 1))
    
    # c = np.zeros(shape=(num_rows + 1))
    # c[-1] = -1
    
    # bounds = [(0, None) for _ in range(num_rows)] + [(None, None)]
    
    # res = linprog(
    #     c=c,
    #     A_ub=A_ub,
    #     b_ub=b_ub,
    #     A_eq=A_eq,
    #     b_eq=b_eq,
    #     bounds=bounds,
    # )
    
    # return res
    
    # A_eq = np.hstack([np.array([[1] * num_cols]).T, matrix.T])
    # A_eq = np.vstack([A_eq, [0] + [1] * num_rows])
    
    # A_ub = np.hstack([np.array([[1] * num_cols]).T, -matrix.T])
    # A_eq = np.ones(shape=(1, num_rows + 1))
    # A_eq[0, 0] = 0
    
    # # b_eq = [0] * num_cols + [1]
    # b_ub = [0] * num_cols 
    # b_eq = [1]
    
    # c = [-1] + [0] * num_rows 

    # bounds = [(None, None)] + [(0.,None) for _ in range(num_rows)]

    # return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # # A_eq = np.hstack([np.array([[1] * num_rows]).T, matrix])
    # # A_eq = np.vstack([A_eq, [0]+[1]*num_cols])
    # A_ub = np.hstack([np.array([[1] * num_rows]).T, -matrix])
    # A_eq = np.ones(shape=(1, num_cols + 1))
    # A_eq[0, 0] = 0
    
    # # b_eq = [0] * num_rows + [1]
    # b_ub = [0] * num_rows 
    # b_eq = [1]

    # c = [-1] + [0] * num_cols
    
    # bounds = [(None, None)] + [(0,1) for _ in range(num_cols)]
    
    # return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

def support_enumeration(matrix: np.array):
    num_rows, num_cols = matrix.shape

    # Generate all non-empty subsets
    all_row_subsets = [np.array(subset) for i in range(1, num_rows + 1)
                for subset in combinations(range(num_rows), i)]
    all_col_subsets = [np.array(subset) for i in range(1, num_cols + 1)
                for subset in combinations(range(num_cols), i)]
    
    eq = []
    for r in all_row_subsets:
        for c in all_col_subsets:
            res = verify_support(matrix, r, c)
            eq.append((res, r, c))
    return eq

def find_correlated_eq(matrix1: np.array, matrix2: np.array):
    A_ub = matrix1.reshape((1,-1))
    print(A_ub)
    
    

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
# m1, m2 = game_of_chicken()
# m1 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
# m2 = -rps_matrix()

# row_strategy = np.array([0.1, 0.2, 0.7])
# column_strategy = np.array([0.3, 0.2, 0.5])
# row_strategy = np.array([0.3, 0.5, 0.2])

if __name__=="__main__":
    matrix_p1 = np.array([[0, -1, 1],
                        [1, 0, -1],
                        [-1, 1, 0]])
    # matrix_p1 = np.array([[0, 0, -10],
    #                     [1, -10, -10],
    #                     [-10, -10, -10]])
    
    matrix1 = np.array([[0,0,-10],
                        [1,-10,-10],
                        [-10,-10,-10]])
    
    matrix2 = matrix1.T
    
    row_supp = [0,1]
    col_supp = [0,1]
    res = verify_support(matrix1, row_supp, col_supp)
    print()
    print()
    print(res)
    
    
    res = verify_support(matrix2, row_supp, col_supp, for_row=False)
    print()
    print(res)
    
    # pd1 = np.array(
    #     [[-2, 0],
    #     [-3, -1]])
    # pd2 = np.array(
    #     [[-2,-3],
    #     [0,-1]]
    # )

    # x = find_correlated_eq(pd1, pd2)
    # print(x)
    # print("---------")
    
    # supp_r = [[0,1]]
    # supp_c = [[0,1,2]]
    
    # for r in supp_r:
    #     for c in supp_c:
    #         x = verify_support(matrix_p1, r, c)
    #         print(f"{r}, {c} - {x}")
            
    

        
    
    # eq = support_enumeration(matrix_p1.T)
    
    # print("---------------")
    # for x in eq:
    #     print(f"{x[0]}, row {x[1]}, col {x[2]}")

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

# rows, cols, cum_regret_row, cum_regret_col, reg_row, reg_col = regret_minimization(m1, m2, 1000, regret_matching)

# explotability_avg, explotability = [], []

# for i in range(len(rows)):
#     row_strat, col_strat = rows[i], cols[i]
#     avg_row, avg_col = average_strat(rows[0:i+1]), average_strat(cols[0:i+1])
    
#     explotability.append(calculate_exploitability(m1, m2, row_strat, col_strat))
#     explotability_avg.append(calculate_exploitability(m1, m2, avg_row, avg_col))

# # plot_two_exploitability(explotability_avg, explotability)
# print(average_strat(rows))
# print(average_strat(cols))
# # print(rows[-1])
# # print(cols[-1])
# v1, v2 = compute_game_value(m1, m2, average_strat(rows), average_strat(cols))
# print(v1, v2)
# print(expl[-1])

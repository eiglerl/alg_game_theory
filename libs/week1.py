import numpy as np


import numpy as np


def compute_non_zero_sum_game_value(matrix1, matrix2, p1_strat, p2_strat):
    # create probability matrix
    prob_matrix = p2_strat @ p1_strat

    # calculate values
    p1_val = np.sum(matrix1 * prob_matrix)
    p2_val = np.sum(matrix2 * prob_matrix)
    return p1_val, p2_val

def evaluate(matrix, row_strategy, column_strategy):
    row_val, col_val = compute_non_zero_sum_game_value(-matrix, matrix, row_strategy, column_strategy)
    return row_val

def best_response_row(matrix, strat):
    expected_payoffs = (strat @ matrix)
    max_payoff = np.argmax(expected_payoffs)
    best_response = create_pure_strategy(len(expected_payoffs), max_payoff)
    return best_response

def best_response_column(matrix, strat):
    expected_payoffs = (matrix @ strat)
    max_payoff = np.argmax(expected_payoffs)
    best_response = create_pure_strategy(len(expected_payoffs), max_payoff)
    return best_response

def create_pure_strategy(len, index):
    return np.array([[1 if i==index else 0 for i in range(len)]])

def best_response_value_row(matrix, row_strategy):
    response = best_response_row(matrix, row_strategy)
    p1_val, p2_val = compute_non_zero_sum_game_value(-matrix, matrix, row_strategy, response.T)
    return p1_val

def best_response_value_column(matrix, column_strategy):
    response = best_response_column(matrix, column_strategy)
    p1_val, p2_val = compute_non_zero_sum_game_value(-matrix, matrix, response, column_strategy)
    return p2_val
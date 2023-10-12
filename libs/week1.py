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

def find_dominated_actions(matrix, axis):
    dominated_actions = []
    # print("axis:", axis)
    for i in range(matrix.shape[axis]):
        for j in range(matrix.shape[axis]):
            if i >= j:
                continue
            # print(np.take(matrix, i, axis=axis), "vs",np.take(matrix, j, axis=axis))
            if np.all(np.take(matrix, i, axis=axis) >= np.take(matrix, j, axis=axis)):
                dominated_actions.append(j)
    return dominated_actions


def find_dominated(matrix1, matrix2):
    dominated_rows = find_dominated_actions(matrix1, axis=0)
    dominated_columns = find_dominated_actions(matrix2, axis=1)
    return dominated_rows, dominated_columns

def iterated_removal_of_dominated_strategies(matrix1, matrix2):
    temp1 = matrix1[:]
    temp2 = matrix2[:]
    while True:
        dominated_rows, dominated_columns = find_dominated(temp1, temp2)
        if len(dominated_rows) + len(dominated_columns) == 0:
            break
    
        non_dominated_mask = np.ones(temp1.shape[0], dtype=bool)
        non_dominated_mask[dominated_rows] = False
        
        temp1 = temp1[non_dominated_mask]
        temp2 = temp2[non_dominated_mask]

        non_dominated_mask = np.ones(temp1.shape[1], dtype=bool)
        non_dominated_mask[dominated_columns] = False

        temp1 = temp1[:,non_dominated_mask]
        temp2 = temp2[:,non_dominated_mask]
        
        
    return temp1, temp2

        

matrix1 = np.array([[13,1,7], [4,3,6], [-1,2,8]])
matrix2 = np.array([[3,4,3], [1,3,2], [9,8,-1]])
print(matrix1)
print(matrix2, "\n")
temp1, temp2 = iterated_removal_of_dominated_strategies(matrix1, matrix2)
print(temp1)
print(temp2)
        

import numpy as np

def compute_non_zero_sum_game_value(matrix1, matrix2, p1_strat, p2_strat):
    # create probability matrix
    prob_matrix = (p2_strat @ p1_strat).T
    # calculate values
    p1_val = np.sum(matrix1 * prob_matrix)
    p2_val = np.sum(matrix2 * prob_matrix)
    return p1_val, p2_val

# bad design
# def best_response(matrix, strat, for_row=True):
#     expected_payoffs = (strat @ matrix) if for_row else (matrix @ strat)
#     axis = 1 if for_row else 0
#     # len = expected_payoffs.shape[1] if for_row else expected_payoffs.shape[0]
#     len = expected_payoffs.shape[axis]
#     # print(f"excepted pay off ","row " if for_row else "col ", expected_payoffs)
#     max_payoff = np.argmax(expected_payoffs, axis=axis)
#     # print(f"max payoff {max_payoff} -> {np.take(expected_payoffs, max_payoff, axis=axis)}")
#     best_response = create_pure_strategy(len, max_payoff)
#     return best_response

def best_response_to_row_player_in_zerosum(matrix: np.array, row_strat: np.array) -> np.array:
    """Assumes the matrix is the utility matrix of column player."""
    return best_response_to_row_player(matrix, row_strat)
def best_response_to_col_player_in_zerosum(matrix: np.array, col_strat: np.array) -> np.array:
    """Assumes the matrix is the utility matrix of column player."""
    return best_response_to_col_player(-matrix, col_strat)


def best_response_to_row_player(matrix: np.array, row_strat: np.array) -> np.array:
    return best_response_strat(matrix, row_strat=row_strat, col_strat=None).T

def best_response_to_col_player(matrix: np.array, col_strat: np.array) -> np.array:
    return best_response_strat(matrix, row_strat=None, col_strat=col_strat)

def best_response_strat(matrix: np.array, row_strat=None, col_strat=None) -> np.array:
    """
    Get best response to either row or column player. Only set row_strat OR col_strat.
    Should not be called. Use best_response_to_row_player or best_response_to_col_player instead.
    """

    if row_strat is not None and col_strat is not None:
        raise NotImplementedError("Both strategies are set.")
    elif row_strat is not None:
        expected_payoffs = row_strat @ matrix
        axis = 1
    elif col_strat is not None:
        expected_payoffs = matrix @ col_strat
        axis = 0
    else:
        raise NotImplementedError("Both strategies are set to None.")

    n = matrix.shape[axis]
    # print(expected_payoffs)
    max_payoff = np.argmax(expected_payoffs, axis=axis)
    return create_pure_strategy(len=n, index=max_payoff)

def best_response_zerosum(matrix, strat):
    if strat.shape[1] == 1:
        return best_response_to_col_player_in_zerosum(matrix, col_strat=strat)
    else:
        return best_response_to_row_player_in_zerosum(matrix, row_strat=strat)

def create_pure_strategy(len, index):
    response = np.array([[1 if i==index else 0 for i in range(len)]])
    return response

def find_dominated_actions(matrix, axis):
    dominated_actions = []
    for i in range(matrix.shape[axis]):
        for j in range(matrix.shape[axis]):
            if i >= j:
                continue
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


# matrix1 = np.array([[13,1,7], [4,3,6], [-1,2,8]])
# matrix2 = np.array([[3,4,3], [1,3,2], [9,8,-1]])

# # after iteration: [[10]], [[4]]
# matrix1 = np.array([[10,5,3], [0,4,6], [2,3,2]])
# matrix2 = np.array([[4,3,2], [1,6,0], [1,5,8]])
# print(matrix1)
# print(matrix2, "\n")
# temp1, temp2 = iterated_removal_of_dominated_strategies(matrix1, matrix2)
# print(temp1)
# print(temp2)

# matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

# column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()
# br = best_response_to_col_player_in_zerosum(matrix, col_strat=column_strategy)
# print(br)
        

def evaluate(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Value of the row play when the row and column player use their respective strategies"""
    row_val, col_val = compute_non_zero_sum_game_value(matrix, -matrix, row_strategy, column_strategy)
    return row_val


def best_response_value_row(matrix: np.array, row_strategy: np.array) -> float:
    """Value of the row player when facing a best-responding column player in a zero-sum game"""
    response = best_response_to_row_player(-matrix, row_strategy)
    p1_val, p2_val = compute_non_zero_sum_game_value(matrix, -matrix, row_strategy, response)
    return p1_val


def best_response_value_column(matrix: np.array, column_strategy: np.array) -> float:
    """Value of the column player when facing a best-responding row player in a zero-sum game"""
    response = best_response_to_col_player(matrix, column_strategy)
    p1_val, p2_val = compute_non_zero_sum_game_value(matrix, -matrix, response, column_strategy)
    return p2_val


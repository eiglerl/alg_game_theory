import numpy as np
import libs.week1 as week1


def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> np.array:
    """Computer how much the players could improve if they were to switch to a best response"""    
    
    br_col = week1.best_response(matrix, column_strategy, for_row=False)
    br_row = week1.best_response(matrix, row_strategy, for_row=True)

    u_row, u_col = week1.compute_non_zero_sum_game_value(matrix, -matrix, row_strategy, column_strategy)
    br_u_col = week1.compute_non_zero_sum_game_value(matrix, -matrix, br_col, column_strategy.T)
    br_u_row = week1.compute_non_zero_sum_game_value(matrix, -matrix, row_strategy, br_row.T)

    return np.array([br_u_row - u_row, br_u_col - u_col])

def nash_conv(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    return sum(compute_deltas(matrix, row_strategy, column_strategy))

def compute_exploitability_zero_sum(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Compute exploitability for a zero-sum game"""
    return nash_conv(matrix, row_strategy, column_strategy)/2


def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Computes epsilon as defined for epsilon-Nash equilibrium"""
    return max(compute_deltas(matrix, row_strategy, column_strategy))


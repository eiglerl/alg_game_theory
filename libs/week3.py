import numpy as np
import week1 as week1


def compute_deltas(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> np.array:
    """Computer how much the players could improve if they were to switch to a best response"""    
    
    best_row_strategy = week1.best_response(matrix, column_strategy, for_row=False)
    best_col_strategy = week1.best_response(-matrix, row_strategy, for_row=True)

    utility_row, utility_col = week1.compute_non_zero_sum_game_value(matrix, -matrix, row_strategy, column_strategy)
    best_row_utility, _ = week1.compute_non_zero_sum_game_value(matrix, -matrix, best_row_strategy, column_strategy)
    _, best_col_utility = week1.compute_non_zero_sum_game_value(matrix, -matrix, row_strategy, best_col_strategy.T)
    print(f"col {best_col_utility}-{utility_col}={best_col_utility-utility_col}")
    print(f"row {best_row_utility}-{utility_row}={best_row_utility-utility_row}")
    return np.array([best_row_utility - utility_row, best_col_utility - utility_col])

def nash_conv(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    return sum(compute_deltas(matrix, row_strategy, column_strategy))

def compute_exploitability_zero_sum(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Compute exploitability for a zero-sum game"""
    return nash_conv(matrix, row_strategy, column_strategy)/2


def compute_epsilon(matrix: np.array, row_strategy: np.array, column_strategy: np.array) -> float:
    """Computes epsilon as defined for epsilon-Nash equilibrium"""
    return max(compute_deltas(matrix, row_strategy, column_strategy))


matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
row_strategy = np.array([[0.1, 0.2, 0.7]])
column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

delta_row, delta_column = compute_deltas(matrix=matrix, row_strategy=row_strategy,
                                                column_strategy=column_strategy)
print(f"row: {delta_row}", f"col: {delta_column}",sep='\n')

    # assert delta_row == pytest.approx(0.12)
    # assert delta_column == pytest.approx(0.68)


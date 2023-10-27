from typing import List, Optional
from scipy.optimize import linprog
import numpy as np
import week1 as week1
# from . import week1
import matplotlib.pyplot as plt

def verify_support_one_side(matrix: np.array, support_row: List, support_col: List) -> Optional[List]:
    """Tries to see whether the column player can mix their strategies in the support so that the values of the row player are best-responding"""
    submatrix = matrix[support_row][:, support_col]
    # print(f"submatrix {submatrix}")
    # print(f"submatrix.T {submatrix.T}")
    result = verify_matrix(submatrix)
    if result.success:
        return result.x[1:]
    return None
    num_rows, num_cols = submatrix.shape
    # print(f"rows: {num_rows}, cols: {num_cols}")



    # 1*-U_1 1*p_1 0*p_2 = 0
    # 1*-U_2 2*p_1 1*p_2 = 0
    # 0 1*p_1 1*p_2 = 1

    
    # print(f"submatrix: \n{submatrix}")
    # print(f"submatrix.T: \n{submatrix.T}")
    # add utility
    A_eq = np.hstack([np.array([[1]*num_rows]).T, submatrix])
    print(A_eq)
    # print(f"with utility: \n{A_eq}")

    # probabilites sum to 1
    A_eq = np.vstack([A_eq, [0]+[1]*(num_cols)])
    # total utility + sum of prob[i] * utility[i] = 0, sum of prob[i] = 1
    b_eq = [0] * (num_rows) + [1]
    # max utility
    c = [1] + [0] * num_cols

    # bounds for utility and probabilities
    bounds = [(None, None)] + [(0,1) for _ in range(num_cols)]



    # print(f"A_eq: {A_eq.shape}")
    # print(f"b_eq: {len(b_eq)}")
    # print(f"c: {len(c)}")
    # print(f"submatrix: \n{A_eq}")
    # print(f"b: {b_eq}")
    # print(f"c: {c}")
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # result = linprog(c, A_ub=A_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if result.success:
        # if np.all(result.x[1:] > 0):
        return result.x[1:]
            # print("more than 0")
        # print(f"utility: {result.x[0]}")
        
    
        return None

    # print(matrix)
    # print(submatrix)
    # print(support_row)
    # print(support_col)
    return None

def verify_matrix(submatrix: np.array):
    num_rows, num_cols = submatrix.shape
    A_eq = np.hstack([np.array([[1]*num_rows]).T, submatrix])
    A_eq = np.vstack([A_eq, [0]+[1]*(num_cols)])
    print(A_eq)
    print('--')
    b_eq = [0] * (num_rows) + [1]
    print(b_eq)
    print('--')
    c = [1] + [0] * num_cols
    print(c)
    bounds = [(None, None)] + [(0,1) for _ in range(num_cols)]

    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)


def get_all_possible_supports(n_actions: int, m_actions: int):
    actions1 = list(range(n_actions))
    actions2 = list(range(m_actions))
    subset_pairs = []
    for i in range(1,1 << len(actions1)):
        subset1 = [actions1[j] for j in range(len(actions1)) if (i & (1 << j)) > 0]

        for j in range(1,1 << len(actions2)):
            subset2 = [actions2[k] for k in range(len(actions2)) if (j & (1 << k)) > 0]

            subset_pairs.append((subset1, subset2))
    return subset_pairs


def nash_equlibrium_for_supports(matrix: np.array):
    n, m = matrix.shape
    all_supports = get_all_possible_supports(n, m)
    results = {}
    i = 1
    for row, col in all_supports:
        # print(f"{i}/{len(all_supports)}")
        i += 1
        res = verify_support_one_side(matrix=matrix, support_row=row, support_col=col)
        if res is not None and len(row) > 1 and len(col) > 1:
            results[f"row: {row}, col: {col}"] = res
    
    return results


def best_response_value_function(matrix: np.array, step_size: float):
    # steps = np.linspace(0, 1, n)
    steps = np.arange(0, 1 + step_size, step_size)
    vals = []
    for step in steps:
        new_strat = np.array([[step, 1-step]])
        br = week1.best_response_to_row_player_in_zerosum(matrix, new_strat)
        # TODO: weird graph
        p1_val, p2_val = week1.compute_non_zero_sum_game_value(matrix, -matrix, new_strat, br)
        vals.append(p1_val)
    
    plt.scatter(steps, vals, s=10)
    plt.show()


# matrix = np.array([[2, 0, 0.8],
#                    [-1, 1, -0.5]])
# strat_col = np.array([[0.2, 0.3, 0.5]])
# strat_row = np.array([[0.4, 0.6]])

# best_response_value_function(matrix, step_size=0.01)

matrix_p1 = np.array([[0, 0, -10],
                      [1, -10, -10],
                      [-10, -10, -10]])
matrix_p2 = np.array([[0, 1, -10],
                      [0, -10, -10],
                      [-10, -10, -10]])


result = verify_support_one_side(matrix = matrix_p1, support_row=[0, 1], support_col = [0, 1, 2])
print(f"result: {result}")
print('------------')
result = verify_support_one_side(matrix = matrix_p1.T, support_row=[0, 1], support_col = [0, 1, 2])
print(f"result: {result}")

# enumerate_all_supports(matrix)
# matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
# results = nash_equlibrium_for_supports(matrix)

# for strat, res in results.items():
#     print(strat, "prob:", res)


# matrix_p1 = np.array([[0, 0, -10], [1, -10, -10], [-10, -10, -10]])
# results = nash_equlibrium_for_supports(matrix_p1)
# for strat, res in results.items():
#     print(strat, "prob:", res)
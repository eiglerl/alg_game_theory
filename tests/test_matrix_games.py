import numpy as np
from libs import matrix_games
import pytest

def test_pure_strat1():
    n = 3
    a = 0
    
    strat = matrix_games.create_pure_strategy(n=n, a=a)
    assert np.array_equal(strat, [1, 0, 0])

def test_pure_strat2():
    n = 3
    a = 1
    strat = matrix_games.create_pure_strategy(n=n, a=a)
    assert np.array_equal(strat, [0, 1, 0])

def test_uniform_strat():
    uniform = matrix_games.create_uniform_strategy(3)
    
    assert np.array_equal(uniform, np.array([1/3, 1/3, 1/3]))

def test_prob_table1():
    strat1 = np.array([1,0,0])
    strat2 = np.array([1,0])
    
    prob_table = matrix_games.create_prob_table(strat1=strat1, strat2=strat2)
    assert np.array_equal(prob_table, np.array([[1,0],[0,0],[0,0]]))

def test_prob_table2():
    strat1 = np.array([0.1,0.4,0.5])
    strat2 = np.array([0.9,0.1])
    
    prob_table = matrix_games.create_prob_table(strat1=strat1, strat2=strat2)
    assert np.allclose(prob_table, np.array([[0.09, 0.01], [0.36, 0.04], [0.45, 0.05]]), atol=1e-8)
    
def test_best_response_rps_against_row():
    matrix = matrix_games.rps_matrix()
    row_strat = np.array([0.1, 0.2, 0.7])
    br = matrix_games.best_response_to_row_player(-matrix, row_strategy=row_strat)
    assert np.array_equal(br, np.array([0, 1, 0]))

def test_best_response_rps_against_column():
    matrix = matrix_games.rps_matrix()
    column_strategy = np.array([0.3, 0.2, 0.5])
    br = matrix_games.best_response_to_column_player(matrix, col_strategy=column_strategy)
    assert np.array_equal(br, np.array([0, 1, 0]))

def test_avg_strat1():
    strats = [matrix_games.create_pure_strategy(3,0), matrix_games.create_pure_strategy(3,1), matrix_games.create_pure_strategy(3,2)]
    avg_strat = matrix_games.average_strat(strats)
    
    assert np.array_equal(avg_strat, np.array([1/3, 1/3, 1/3]))
    
def test_avg_strat2():
    strats = [matrix_games.create_pure_strategy(3,0), matrix_games.create_pure_strategy(3,0), matrix_games.create_pure_strategy(3,0), matrix_games.create_pure_strategy(3,2)]
    avg_strat = matrix_games.average_strat(strats)
    
    assert np.array_equal(avg_strat, np.array([3/4, 0, 1/4]))

def test_best_response_to_average_strat():
    matrix = matrix_games.rps_matrix()
    
    strats = [np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.3, 0.]), np.array([0.1, 0.7, 0.2]), np.array([0.4, 0.5, 0.1])]
    br_to_row = matrix_games.best_response_to_average_row_strat(-matrix, strats)
    br_to_col = matrix_games.best_response_to_average_col_strat(matrix, strats)
    
    assert np.array_equal(br_to_row, np.array([1,0,0]))
    assert np.array_equal(br_to_col, np.array([1,0,0]))

def test_best_response_to_last_strat():
    matrix = matrix_games.rps_matrix()

    strats = [np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.3, 0.]), np.array([0.1, 0.7, 0.2]), np.array([0.4, 0.5, 0.1])]
    br_to_row = matrix_games.best_reponse_to_last_row_strat(-matrix, strats)
    br_to_col = matrix_games.best_reponse_to_last_col_strat(matrix, strats)

    assert np.array_equal(br_to_row, np.array([1,0,0]))
    assert np.array_equal(br_to_col, np.array([1,0,0]))

def test_deltas():
    matrix = matrix_games.rps_matrix()
    row_strategy = np.array([0.1, 0.2, 0.7])
    column_strategy = np.array([0.3, 0.2, 0.5])

    delta_row, delta_column = matrix_games.calculate_deltas(matrix, -matrix, row_strategy, column_strategy)
    assert delta_row == pytest.approx(0.12)
    assert delta_column == pytest.approx(0.68)




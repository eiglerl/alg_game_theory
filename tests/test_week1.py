import libs.week1 as week1
import numpy as np
import pytest


def test_week1():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    row_strategy = np.array([[0.1, 0.2, 0.7]])
    column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

    row_value = week1.evaluate(matrix=matrix, row_strategy=row_strategy, column_strategy=column_strategy)
    assert row_value == pytest.approx(0.08)

    br_value_row = week1.best_response_value_row(matrix=matrix, row_strategy=row_strategy)
    br_value_column = week1.best_response_value_column(matrix=matrix, column_strategy=column_strategy)
    assert br_value_row == pytest.approx(-0.6)
    assert br_value_column == pytest.approx(-0.2)


def test_prob():
    matrix = np.array([[0, 1, -1], [-1, 0, 1]])

    row_strat = np.array([[0.1, 0.9]])
    col_strat = np.array([[0.3, 0.2, 0.5]]).T

    prob_matrix = (col_strat @ row_strat).T#.reshape(matrix.shape)
    print(prob_matrix)
    utility = prob_matrix * matrix

    print(utility)
    print("-----")

    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    row_strat = np.array([[0.1, 0.7, 0.2]])
    col_strat = np.array([[0.3, 0.2, 0.5]]).T

    prob_matrix = (col_strat @ row_strat).T
    print(prob_matrix)
    utility = prob_matrix * matrix

    print(utility)

# best response in RPS
def test_best_response_rps_against_row():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    row_strat = np.array([[0.1, 0.2, 0.7]])
    br = week1.best_response_to_row_player_in_zerosum(matrix, row_strat=row_strat)
    assert np.array_equal(br, np.array([[1, 0, 0]]).T)

def test_best_response_rps_against_column():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    column_strategy = np.array([[0.3, 0.2, 0.5]]).T
    br = week1.best_response_to_col_player_in_zerosum(matrix, col_strat=column_strategy)
    assert np.array_equal(br, np.array([[1, 0, 0]]))

# MxN matrix
def test_best_response_nxm_against_row():
    matrix = np.array([[1, 0], [-1, 1], [1, 0]])
    row_strat = np.array([[0.1, 0.2, 0.7]])
    br = week1.best_response_to_row_player(matrix, row_strat=row_strat)
    assert np.array_equal(br, np.array([[1, 0]]).T)
def test_best_response_nxm_against_column():
    matrix = np.array([[1, 0], [-1, 1], [1, 0]])
    column_strategy = np.array([[0.3, 0.7]]).T
    br = week1.best_response_to_col_player(matrix, col_strat=column_strategy)
    assert np.array_equal(br, np.array([[0, 1, 0]]))

def test_iterated_removal():
    matrix1 = np.array([[13,1,7], [4,3,6], [-1,2,8]])
    matrix2 = np.array([[3,4,3], [1,3,2], [9,8,-1]])

    matrixA, matrixB = week1.iterated_removal_of_dominated_strategies(matrix1=matrix1, matrix2=matrix2)
    expected_matrixA = np.array([[13, 1], [4, 3]])
    expected_matrixB = np.array([[3, 4], [1, 3]])

    assert np.array_equal(matrixA, expected_matrixA)
    assert np.array_equal(matrixB, expected_matrixB)

    matrix1 = np.array([[10,5,3], [0,4,6], [2,3,2]])
    matrix2 = np.array([[4,3,2], [1,6,0], [1,5,8]])

    matrixA, matrixB = week1.iterated_removal_of_dominated_strategies(matrix1=matrix1, matrix2=matrix2)
    expected_matrixA = np.array([[10]])
    expected_matrixB = np.array([[4]])

    assert np.array_equal(matrixA, expected_matrixA)
    assert np.array_equal(matrixB, expected_matrixB)

test_prob()
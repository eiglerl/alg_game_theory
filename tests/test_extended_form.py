import numpy as np
from libs import extensive_form_games
import pytest

def test_reach_probability_rps_uniform():
    game = extensive_form_games.rps()
    actions = ["R", "P", "S"]
    histories = [p1+p2 for p1 in actions for p2 in actions]
    p1_strat = game.uniform_strat_for_player("Player1")
    p2_strat = game.uniform_strat_for_player("Player2")
    strategies = p1_strat
    strategies.update(p2_strat)
    
    probabilities = [game.calculate_reach_probability(strategies, h) for h in histories]
    expected = [1/len(histories) for _ in range(len(histories))]
    assert np.allclose(probabilities, expected, atol=1e-8)
    
def test_reach_probability_simple_poker_uniform():
    game = extensive_form_games.kuhn_poker()
    p1_strat = game.uniform_strat_for_player("Player1")
    p2_strat = game.uniform_strat_for_player("Player2")
    chance_strat = game.uniform_strat_for_player("Chance")
    
    strategies = p1_strat
    strategies.update(p2_strat)
    strategies.update(chance_strat)
    
    cards = ["A", "K"]
    chance_histories = [c1+c2 for c1 in cards for c2 in cards]
    probabilities = [game.calculate_reach_probability(strategies, h) for h in chance_histories]
    expected = [1/len(chance_histories) for _ in range(len(chance_histories))]
    
    assert np.allclose(probabilities, expected, atol=1e-8)
    
    chosen_cards = "AA"
    histories = ["CBC", "CBF", "BC", "BF"]
    histories = [chosen_cards+h for h in histories]
    expected = [1/2**(len(h)) for h in histories]
    probabilities = [game.calculate_reach_probability(strategies, h) for h in histories]
    
    assert np.allclose(probabilities, expected, atol=1e-8)
    
def test_reach_probability_simple_poker():
    game = extensive_form_games.kuhn_poker()
    strategies = {
        "Player1": {
            "A": {
                "C": 0.9,
                "B": 0.1
            },
            "ACB": {
                "C": 0.9,
                "F": 0.1
            }
        },
        "Player2": {
            "AC": {
                "B": 0.7,
                "C": 0.3
            },
            "AB": {
                "C": 0.7,
                "F": 0.3
            }
        },
        "Chance": {
            "": {
                "A": 0.5,
                "K": 0.5
            }
        }
    }
    p1_strat = strategies["Player1"]
    p2_strat = strategies["Player2"]
    
    histories = ["AACBC", "AACBF", "AACC", "AABC", "AABF"]

    chosen_cards_prob = 0.5*0.5
    expected = [
        p1_strat["A"]["C"]*p2_strat["AC"]["B"]*p1_strat["ACB"]["C"],
        p1_strat["A"]["C"]*p2_strat["AC"]["B"]*p1_strat["ACB"]["F"],
        p1_strat["A"]["C"]*p2_strat["AC"]["C"],
        p1_strat["A"]["B"]*p2_strat["AB"]["C"],
        p1_strat["A"]["B"]*p2_strat["AB"]["F"],
    ]
    expected = [chosen_cards_prob * p for p in expected]
    
    probabilities = [game.calculate_reach_probability(strategies, h) for h in histories]
    assert np.allclose(probabilities, expected, atol=1e-8)

def test_deltas_rps():
    game = extensive_form_games.rps()
    p1_strat = {
        "Player1": {
            "": {
                "R": 1,
                "P": 0,
                "S": 0
            }
        }
    }
    p2_strat = {
        "Player2": {
            "": {
                "R": 0,
                "P": 1,
                "S": 0
            }
        }
    }
    
    strategies = p1_strat
    strategies.update(p2_strat)
    
    # u_i(br(pi_-i), pi_-i) - u_i(pi)
    exptected_deltas = {
        "Player1": 1 - (-1),
        "Player2": 1 - 1
    }
    
    deltas = game.calculate_deltas(strategies)
    
    # Convert dictionaries to arrays
    array1 = np.array(list(exptected_deltas.items()))
    array2 = np.array(list(deltas.items()))

    # Sort arrays to ensure the order of items doesn't affect the comparison
    array1.sort(axis=0)
    array2.sort(axis=0)
    
    assert np.array_equal(array1, array2)
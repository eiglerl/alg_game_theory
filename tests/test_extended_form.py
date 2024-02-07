import numpy as np
from libs import extensive_form_games
import pytest

def assert_dicts_equal(dict1: dict, dict2: dict):
    for key in dict1.keys():
        if key not in dict2:
            raise AssertionError(f"Key '{key}' not found in the second dictionary")

        value1, value2 = dict1[key], dict2[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            # Recursively check nested dictionaries
            assert_dicts_equal(value1, value2)
        else:
            assert dict1[key] == dict2[key] and not isinstance(value1, dict) and not isinstance(value2, dict)

def avg_example_game() -> extensive_form_games.ExtensiveFormGame:
    root = extensive_form_games.Node("Player1", "", 
    {
        "L": extensive_form_games.Node("Player2", "L", 
        {
            "L": extensive_form_games.Node("", "LL"),
            "R": extensive_form_games.Node("", "LR")
        }),
        "R": extensive_form_games.Node("Player2", "R", 
        {
            "L": extensive_form_games.Node("", "RL"),
            "R": extensive_form_games.Node("Player1", "RR", 
            {
                "L": extensive_form_games.Node("", "RRL"),
                "R": extensive_form_games.Node("", "RRR")
            })
        })
    })
    matrix = {
        "LL": {"Player1": -2, "Player2": 2},
        "LR": {"Player1": 1, "Player2": -1},
        "RL": {"Player1": 1, "Player2": -1},
        "RRL": {"Player1": 4, "Player2": -4},
        "RRR": {"Player1": 0, "Player2": 0},
    }
    tree = extensive_form_games.ExtensiveFormGame(["Player1", "Player2"], [], root, matrix)
    return tree


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


def test_deltas_rps_simple():
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
    
    # delta_i = u_i(br(pi_-i), pi_-i) - u_i(pi)
    exptected_deltas = {
        "Player1": 1 - (-1),
        "Player2": 1 - 1
    }
    
    deltas = game.calculate_deltas(strategies)
    
    assert_dicts_equal(exptected_deltas, deltas)

def test_average_strat():
    game = avg_example_game()
    
    s1 = {
        "Player1": {
            "": {
                "L": 0.2,
                "R": 0.8
            },
            "RR": {
                "L": 0.8,
                "R": 0.2
            }
        }
    }
    s2 = {
        "Player1": {
            "": {
                "L": 0.8,
                "R": 0.2
            },
            "RR": {
                "L": 0.2,
                "R": 0.8
            }
        }
    }
    
    avg = game.average_strategy([s1, s2])
    expected = {
        "Player1": {
            "": {
                "L": 0.5,
                "R": 0.5
            },
            "RR": {
                "L": 0.68,
                "R": 0.32
            }
        }
    }
    
    assert_dicts_equal(expected, avg)
    


def test_best_response_rps():
    game = extensive_form_games.rps()
    strategies = {
        "Player1": {
            "": {
                "R": 1,
                "P": 0,
                "S": 0
            }
        }
    }
    expected = {
        "Player2": {
            "": {
                "R": 0,
                "P": 1,
                "S": 0
            }
        }
    }
    
    br = game.best_response(strategies)
    
    assert_dicts_equal(expected, br)
    
    # NOT FINISHED
def test_best_response_simple_poker():
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
            },
            "K": {
                "C": 0.1,
                "B": 0.9
            },
            "KCB": {
                "C": 0.1,
                "F": 0.9
            }
        },
        "Chance": {
            "": {
                "A": 0.5,
                "K": 0.5
            }
        }
    }
    
    br = game.best_response(strategies)
    print(br)
    
    strategies.update(br)
    
    deltas = game.calculate_deltas(strategies)
    print(br)
    assert False

def test_regret_matching():
    def check_uniform(strategy, player):
        for k,v in strategy[player][''].items():
            assert np.isclose(v, 1/3, atol=1e-8)

    p="Player1"
    
    tree = extensive_form_games.rps()
    regrets = tree._prepare_regrets(p)
    strat = tree.regret_matching_cfr(regrets, p)
    
    # All zero
    check_uniform(strat,p)
    
    # One negative
    regrets['']["R"] = -1
    strat = tree.regret_matching_cfr(regrets, p)
    check_uniform(strat,p)
    
    # One positive
    regrets['']["R"] = 150
    strat = tree.regret_matching_cfr(regrets, p)
    assert strat[p]['']["R"] == 1 and strat[p]['']["P"] == 0 and strat[p]['']["S"] == 0
    
    # All positive
    regrets['']["R"] = 1
    regrets['']["P"] = 2
    regrets['']["S"] = 3
    strat = tree.regret_matching_cfr(regrets, "Player1")
    assert np.isclose(strat[p]['']["R"],1/6,atol=1e-8) and \
        np.isclose(strat[p]['']["P"], 1/3, atol=1e-8) \
            and np.isclose(strat[p]['']["S"],1/2,atol=1e-8)

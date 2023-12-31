import numpy as np

class Node:
    def __init__(self, player, information_set, children=None, history=None):
        self.player = player

        self.information_set = information_set
        self.history = history or information_set
        self.children = children or {}
    
    def is_terminal(self):
        return not self.children
    
    def actions(self):
        return self.children.keys()    

class ExtensiveFormGame:
    def __init__(self, players: list, chance: list, root: Node, matrix: dict):
        self.players = players
        self.chance = chance
        self.root = root
        self.matrix = matrix
        self.infosets = self._prepare_infosets()
    
    def get_infosets(self) -> dict:
        return self.infosets
    
    def get_nodes_in_infoset(self, infoset, player=None) -> set:
        if player is not None:
            return set(n for n in self.infosets[infoset] if n.player == player)
        return self.infosets[infoset]
    
    def print_tree(self, node: Node = None, indent=""):
        node = node or self.root
        value = f" {self.matrix[node.information_set]}" if node.information_set in self.matrix else ""
        print(f"{indent}{node.history}{value}")

        for child_action, child_node in node.children.items():
            self.print_tree(child_node, indent + "  ")
        
    
    def _prepare_infosets(self) -> dict:
        infosets = {}
        
        self._dfs_infosets(self.root, infosets)
        return infosets
        
    def _dfs_infosets(self, node: Node, infosets: dict):
        infosets.setdefault(node.information_set, set())
        infosets[node.information_set].add(node)
        for k, child in node.children.items():
            self._dfs_infosets(child, infosets)
    
    def calculate_reach_probability(self, strategies, history):
        node = self.root
        prob = 1
        
        for a in history:
            if node.player in strategies:
                prob *= strategies[node.player][node.information_set][a]
            node = node.children[a]
        
        return prob
    
    def calculate_probabilities(self, strategies, node: Node =None, prob=1):
        node = node or self.root
        
        if node.is_terminal():
            print(f'{node.information_set} with prob {prob}, value {self.matrix[node.information_set]}')
            return
        
        player = node.player
        information_set = node.information_set
        
        for action in node.actions():
            prob_of_action = strategies[player][information_set][action]
            child = node.children[action]
            
            self.calculate_probabilities(strategies, child, prob*prob_of_action)
    
    def get_node_by_history(self, history):
        node = self.root
        for a in history:
            node = node.children[a]
        return node
    
    def calculate_player_values(self, strategies, node: Node =None):
        node = node or self.root
        
        return self.history_value(strategies, node)
        
    def history_value(self, strategies, node: Node):
        if node.is_terminal():
            return self.matrix[node.history]
        
        player = node.player
        information_set = node.information_set
        
        values = {p: 0 for p in self.players}

        for action in node.actions():
            a_values = self.history_action_value(strategies, node, action)
            for p in self.players:
                values[p] += a_values[p] * strategies[player][information_set][action]

        return values
    
    def history_action_value(self, strategies, node, action):
        return self.history_value(strategies, node.children[action])
    
    def find_all_histories(self, node: Node = None, history=[]):
        node = node or self.root
        histories = set()

        if node.is_terminal():
            histories.add(tuple(history))

        for action in node.actions():
            new_history = history[:]
            new_history.append(str(action))
            histories.update(
                self.find_all_histories(node.children[action], new_history)
            )

        return histories
    
    def average_strategy(self, strategies):
        player = next(iter(strategies[0]))
        avg_strat = {player: {}}
        for information_set in strategies[0][player].keys():
            avg_strat[player][information_set] = {}
            for action in strategies[0][player][information_set].keys():
                avg_strat[player][information_set][action] = 0
        # print(avg_strat)
        self._avg_strat_dfs(self.root, strategies, avg_strat, player)
        
        # normalize
        for k in avg_strat[player].keys():
            total_sum = sum(avg_strat[player][k].values())
            for action in avg_strat[player][k]:
                avg_strat[player][k][action] /= total_sum
        
        return avg_strat
    
    def _avg_strat_dfs(self, node: Node, strategies, avg_strat, player):
        if node.is_terminal():
            return
        
        if node.player != player:
            for action in node.actions():
                self._avg_strat_dfs(node.children[action], strategies, avg_strat, player)
            return
        
        information_set = node.information_set
        reach_prob = self.calculate_reach_probability(avg_strat, node.history)
        strat_reach_prob = [self.calculate_reach_probability(s, node.history) for s in strategies]
        
        for action in node.actions():
            for i, strat in enumerate(strategies):
                avg_strat[player][information_set][action] += strat_reach_prob[i] * strat[player][information_set][action]

            self._avg_strat_dfs(node.children[action], strategies, avg_strat, player)
                
        
                
    def best_response(self, strategies, node: Node = None):
        assert len(strategies.keys()) == len(self.players) - 1
        player = next(p for p in self.players if p not in strategies)
        node = node or self.root
        
        best_response = {}
        self._find_best_response(strategies, node, best_response, player)
        
        return best_response
    
    def _find_best_response(self, strategies, node: Node, best_response: dict, player, br_values: dict = {}):
        if node.is_terminal():
            return self.matrix[node.information_set][player]
        
        action_values = {}
        if node.player == player:
            nodes = self.get_nodes_in_infoset(node.information_set, player)
            reach_probabilities = [self.calculate_reach_probability(strategies, n.history) for n in nodes]
            action_values = {}
            best_response[node.information_set] = {}
            for action in node.actions():
                action_values[action] = 0
                best_response[node.information_set][action] = 0
                
                for i, possible_node in enumerate(nodes):
                    action_values[action] += \
                        reach_probabilities[i] * \
                        self._find_best_response(strategies, possible_node.children[action], best_response, player, br_values)
                # action_values.append(self._find_best_response(strategies, node.child, best_response, player, br_values))
            
            best_response[node.information_set][max(action_values, key=action_values.get)] = 1
        else:
            for action in node.actions():
                action_values[action] = \
                    strategies[node.player][node.information_set][action] * \
                    self._find_best_response(strategies, node.children[action], best_response, player, br_values)

        return max(action_values.values())
        


    

def rps() -> ExtensiveFormGame:

    root = Node("Player1", "", 
    {
        "R": Node("Player2", "", {
            "R": Node("", "RR"),
            "P": Node("", "RP"),
            "S": Node("", "RS"),
        }, history="R"),
        "P": Node("Player2", "", {
            "R": Node("", "PR"),
            "P": Node("", "PP"),
            "S": Node("", "PS"),
        }, history="P"),
        "S": Node("Player2", "", {
            "R": Node("", "SR"),
            "P": Node("", "SP"),
            "S": Node("", "SS"),
        }, history="S")
        
    })
    matrix = {
        "RR": {"Player1": 0, "Player2": 0},
        "RP": {"Player1": -1, "Player2": 1},
        "RS": {"Player1": 1, "Player2": -1},
        "PR": {"Player1": 1, "Player2": -1},
        "PP": {"Player1": 0, "Player2": 0},
        "PS": {"Player1": -1, "Player2": 1},
        "SR": {"Player1": -1, "Player2": 1},
        "SP": {"Player1": 1, "Player2": -1},
        "SS": {"Player1": 0, "Player2": 0},
    }
    tree = ExtensiveFormGame(["Player1", "Player2"], [], root, matrix)
    return tree

def kuhn_poker():
    root = Node("Chance", "", {
        "K": Node("Chance", "", {
                "K": Node("Player1", "K",{
                    "C": Node("Player2", "KC", {
                        "B": Node("Player1", "KCB",{
                            "C": Node("", "KKCBC"),
                            "F": Node("", "KKCBF")
                            }, "KKCB"),
                        "C": Node("", "KKCC")
                        }, "KKC"),
                    "B": Node("Player2", "KB", {
                        "C": Node("", "KKBC"),
                        "F": Node("", "KKBF")
                        }, "KKB")
                }, "KK"),
                "A": Node("Player1", "K",{
                    "C": Node("Player2", "AC", {
                        "B": Node("Player1", "KCB", {
                            "C": Node("", "KACBC"),
                            "F": Node("", "KACBF")
                            }, "KACB"),
                        "C": Node("", "KACC")
                        }, "KAC"),
                    "B": Node("Player2", "AB", {
                        "C": Node("", "KABC"),
                        "F": Node("", "KABF")
                    })
                }, "KA")
            }, "K"),
        
        "A": Node("Chance", "", {
                "K": Node("Player1", "A",{
                    "C": Node("Player2", "KC", {
                        "B": Node("Player1", "ACB", {
                            "C": Node("", "AKCBC"),
                            "F": Node("", "AKCBF")
                            }, "AKCB"),
                        "C": Node("", "AKCC")
                        }, "AKC"),
                    "B": Node("Player2", "KB", {
                        "C": Node("", "AKBC"),
                        "F": Node("", "AKBF")
                    }, "AKB")
                }, "AK"),
                "A": Node("Player1", "A",{
                    "C": Node("Player2", "AC", {
                        "B": Node("Player1", "ACB", {
                            "C": Node("", "AACBC"),
                            "F": Node("", "AACBF")
                            }, "AACB"),
                        "C": Node("", "AACC")
                        }, "AAC"),
                    "B": Node("Player2", "AB", {
                        "C": Node("", "AABC"),
                        "F": Node("", "AABF")
                    })
                }, "AA")
            }, "A")
    })
    
    matrix = {
        "KKCBC": {"Player1": 0, "Player2": 0},
        "KKCBF": {"Player1": -2, "Player2": 2},
        "KKCC": {"Player1": 0, "Player2": 0},
        "KKBC": {"Player1": 0, "Player2": 0},
        "KKBF": {"Player1": -2, "Player2": 2},
        
        "KACBC": {"Player1": -3, "Player2": 3},
        "KACBF": {"Player1": -2, "Player2": 2},
        "KACC": {"Player1": -2, "Player2": 2},
        "KABC": {"Player1": -3, "Player2": 3},
        "KABF": {"Player1": 2, "Player2": -2},
        
        "AKCBC": {"Player1": 3, "Player2": -3},
        "AKCBF": {"Player1": -2, "Player2": 2},
        "AKCC": {"Player1": 2, "Player2": -2},
        "AKBC": {"Player1": 3, "Player2": -3},
        "AKBF": {"Player1": 2, "Player2": -2},
        
        "AACBC": {"Player1": 0, "Player2": 0},
        "AACBF": {"Player1": -2, "Player2": 2},
        "AACC": {"Player1": 0, "Player2": 0},
        "AABC": {"Player1": 0, "Player2": 0},
        "AABF": {"Player1": 2, "Player2": -2}
    }
    tree = ExtensiveFormGame(["Player1", "Player2"], ["Chance"], root, matrix)
    return tree    
    

def avg_example():
    root = Node("Player1", "", {
        "A": Node("Player2", "A", {
            "1": Node("", "A1"),
            "2": Node("", "A2")
        }),
        "B": Node("Player2", "B", {
            "1": Node("", "B1"),
            "2": Node("Player1", "B2", {
                "A": Node("", "B2A"),
                "B": Node("", "B2B")
            })
        })
    })
    
    matrix = {
        "A1": {"Player1": -2, "Player2": 2},
        "A2": {"Player1": 1, "Player2": -1},
        "B1": {"Player1": 1, "Player2": -1},
        "B2A": {"Player1": 4, "Player2": -4},
        "B2B": {"Player1": 0, "Player2": 0},
    }
    tree = ExtensiveFormGame(["Player1", "Player2"], [], root, matrix)
    return tree

# tree = rps()
# opponent_strategies = {
#     "Player1": {
#         "": {
#             "R": 0.3,
#             "P": 0.1,
#             "S": 0.6
#         }
#     }
# }
# tree = kuhn_poker()
# strategies = {
#     "Player1": {
#         "A": {
#             "C": 0.5,
#             "B": 0.5
#         },
#         "K": {
#             "C": 0.5,
#             "B": 0.5
#         },
#         "KCB":{
#             "C": 0.5,
#             "F": 0.5
#         },
#         "ACB":{
#             "C": 0.5,
#             "F": 0.5
#         }
#     },
#     "Player2": {
#         "AB": {
#             "C": 0.5,
#             "F": 0.5
#         },
#         "AC": {
#             "C": 0.5,
#             "B": 0.5
#         },
#         "KB": {
#             "C": 0.5,
#             "F": 0.5
#         },
#         "KC": {
#             "C": 0.5,
#             "B": 0.5
#         },
#     },
#     "Chance": {
#         "": {
#             "K": 0.5,
#             "A": 0.5
#         },
#         "A": {
#             "K": 0.5,
#             "A": 0.5
#         },
#         "K": {
#             "K": 0.5,
#             "A": 0.5
#         }
#     }
# }

# strategies2 = {
#     "Player1": {
#         "A": {
#             "C": 0.9,
#             "B": 0.1
#         },
#         "K": {
#             "C": 0.1,
#             "B": 0.9
#         },
#         "KCB":{
#             "C": 0.4,
#             "F": 0.6
#         },
#         "ACB":{
#             "C": 0.3,
#             "F": 0.7
#         }
#     },
#     "Player2": {
#         "AB": {
#             "C": 0.5,
#             "F": 0.5
#         },
#         "AC": {
#             "C": 0.4,
#             "B": 0.6
#         },
#         "KB": {
#             "C": 0.7,
#             "F": 0.3
#         },
#         "KC": {
#             "C": 0.2,
#             "B": 0.8
#         },
#     },
#     "Chance": {
#         "": {
#             "K": 0.5,
#             "A": 0.5
#         },
#         "A": {
#             "K": 0.5,
#             "A": 0.5
#         },
#         "K": {
#             "K": 0.5,
#             "A": 0.5
#         }
#     }
# }

# list_strat = [{"Player1": strategies["Player1"]}, {"Player1": strategies2["Player1"]}]

# print(tree.find_all_histories())
# tree.print_tree()
# values = tree.calculate_player_values(strategies)
# print(values)
# avg_strat = tree.average_strategy(list_strat)
# print(avg_strat)

# prob = tree.calculate_reach_probability(opponent_strategies, ('R', 'R')) 
# print(prob)
# br = tree.best_response(opponent_strategies)
# print(br)
# for k in tree.infosets.keys():
#     print(f"{k}, len {len(tree.infosets[k])}")
# strategies = {
#     "Player1":{
#         "": {
#             "R": 1/3,
#             "P": 1/3,
#             "S": 1/3
#         }
#     },
#         "Player2":{
#         "": {
#             "R": 1/3,
#             "P": 1/3,
#             "S": 1/3
#         }
#     }
# }
# # tree.calculate_probabilities(strategies)
# val = tree.calculate_player_values(strategies)
# print(val)
# histories = tree.find_all_histories()
# print(sorted(histories))


tree = avg_example()
list_strat = [
    {"Player1":
        {
            "": {"A": 0.2, "B": 0.8},
            "B2": {"A": 0.8, "B": 0.2}
        }
    },
    {"Player1":
        {
            "": {"A": 0.8, "B": 0.2},
            "B2": {"A": 0.2, "B": 0.8}
        }
    },
]

print("Strat1")
print(list_strat[0])
print("Strat2")
print(list_strat[1])
avg_strat = tree.average_strategy(list_strat)
print("avg strat")
print(avg_strat)
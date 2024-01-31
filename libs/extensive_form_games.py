import numpy as np
import matplotlib.pyplot as plt
import copy

class Node:
    def __init__(self, player, information_set, children=None, history=None):
        # Initialize a Node with player, information_set, children, and history attributes
        self.player = player  # Player associated with the node
        self.information_set = information_set  # Information set associated with the node
        self.history = history or information_set  # History of the node, defaults to information_set if not provided
        self.children = children or {}  # Dictionary to store children nodes, defaults to an empty dictionary if not provided
    
    def is_terminal(self):
        # Check if the node is a terminal node (has no children)
        return not self.children
    
    def actions(self):
        # Get the actions available at this node (keys of the children dictionary)
        return self.children.keys()

class ExtensiveFormGame:
    def __init__(self, players: list, chance: list, root: Node, matrix: dict):
        # Initialize an ExtensiveFormGame with players, chance, root node, matrix, infosets, and actions_in_infoset
        self.players = players  # List of players in the game
        self.chance = chance  # List of players representing chance events
        self.root = root  # Root node of the game tree
        self.matrix = matrix  # Dictionary representing the matrix of payoffs
        self.infosets = self._prepare_infosets()  # Dictionary to store information sets
        self.actions_in_infoset = self._prepare_actions_in_infosets()  # Dictionary to store actions in each information set
    
    def get_infosets(self) -> dict:
        # Get the dictionary of information sets
        return self.infosets
    
    def get_nodes_in_infoset(self, infoset, player=None) -> set:
        # Get the set of nodes in a specific information set for a given player (if specified)
        if player is not None:
            return set(n for n in self.infosets[infoset] if n.player == player)
        return self.infosets[infoset] 
        
    def print_tree(self, node: Node = None, indent=""):
        # Print the game tree starting from a specified node (default is the root)
        node = node or self.root
        value = f" {self.matrix[node.information_set]}" if node.information_set in self.matrix else ""
        print(f"{indent}{node.history}{value}")

        for child_action, child_node in node.children.items():
            self.print_tree(child_node, indent + "  ")
        
    def _prepare_infosets(self) -> dict:
        # Prepare and return a dictionary of information sets
        infosets = {}
        
        self._dfs_infosets(self.root, infosets)
        return infosets
        
    def _dfs_infosets(self, node: Node, infosets: dict):
        # Depth-first search to populate the dictionary of information sets
        infosets.setdefault(node.information_set, set())
        infosets[node.information_set].add(node)
        for k, child in node.children.items():
            self._dfs_infosets(child, infosets)
    
    def _prepare_actions_in_infosets(self):
        # Prepare and return a dictionary of actions in each information set
        actions_in_infoset = {}
        for infoset in self.infosets:
            actions_in_infoset[infoset] = next(iter(self.get_nodes_in_infoset(infoset))).actions()
        return actions_in_infoset
        
    def calculate_reach_probability(self, strategies, history):
        # Calculate the reach probability of a given history based on player strategies
        # Player's strategy can be ommited
        node = self.root
        prob = 1
        
        for a in history:
            if node.player in strategies:
                prob *= strategies[node.player][node.information_set][a]
            node = node.children[a]
        
        return prob
    
    def calculate_reach_probability_infoset(self, strategies, infoset, player=None):
        # Calculate the total reach probability for all nodes in a given information set
        prob = 0
        for node in self.get_nodes_in_infoset(infoset, player):
            prob += self.calculate_reach_probability(strategies, node.history)
        return prob
    
    def calculate_probabilities(self, strategies, node: Node = None, prob=1):
        # Calculate and print the probabilities for each node in the game tree based on player strategies
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
        # Get the node in the game tree corresponding to a given history
        node = self.root
        for a in history:
            node = node.children[a]
        return node
    
    def calculate_player_values(self, strategies, node: Node = None):
        # Calculate the player values for a given node in the game tree based on player strategies
        node = node or self.root
        
        return self.history_value(strategies, node)
        
    def history_value(self, strategies, node: Node):
        # Calculate the values for all players at a given node in the game tree based on player strategies
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
        # Calculate the values for all players after taking a specific action at a given node
        return self.history_value(strategies, node.children[action])
    
    def find_all_histories(self, node: Node = None, history=[]):
        # Find all possible histories in the game tree starting from a given node
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
        # Calculate the average strategy over all players and information sets
        player = next(iter(strategies[0]))
        avg_strat = {player: {}}
        for information_set in strategies[0][player].keys():
            avg_strat[player][information_set] = {}
            for action in strategies[0][player][information_set].keys():
                avg_strat[player][information_set][action] = 0
        
        # Use DFS to calculate average strategy
        self._avg_strat_dfs(self.root, strategies, avg_strat, player)
        
        # Normalize the average strategy
        for infoset in avg_strat[player].keys():
            total_sum = sum(avg_strat[player][infoset].values())
            number_of_actions = len(avg_strat[player][infoset].keys())
            for action in avg_strat[player][infoset]:
                if total_sum == 0:
                    avg_strat[player][infoset][action] = 1/number_of_actions
                else:
                    avg_strat[player][infoset][action] /= total_sum
        
        return avg_strat
    
    def _avg_strat_dfs(self, node: Node, strategies, avg_strat, player):
        # Depth-first search to calculate average strategy
        if node.is_terminal():
            return
        
        # Skip nodes where the player does not play
        if node.player != player:
            for action in node.actions():
                self._avg_strat_dfs(node.children[action], strategies, avg_strat, player)
            return
        
        information_set = node.information_set
        # reach_prob = self.calculate_reach_probability(avg_strat, node.history)
        strat_reach_prob = [self.calculate_reach_probability(s, node.history) for s in strategies]
        
        for action in node.actions():
            for i, strat in enumerate(strategies):
                # Sum up (reach probability * probability of action) for each strategy
                avg_strat[player][information_set][action] += strat_reach_prob[i] * strat[player][information_set][action]

            self._avg_strat_dfs(node.children[action], strategies, avg_strat, player)
                
    def best_response(self, strategies, node: Node = None):
        # Find the best response strategy against given strategies
        assert len(strategies.keys()) == len(self.players) + len(self.chance) - 1
        
        # Get the player we need
        player = next(p for p in self.players if p not in strategies)
        node = node or self.root
        
        best_response = {player: {}}
        self._find_best_response(strategies, node, best_response, player)
        
        return best_response
    
    def _find_best_response(self, strategies, node: Node, best_response: dict, player, information_set_values: dict = {}):
        # Use recursive DFS to find the best response strategy
        if node.is_terminal():
            return self.matrix[node.history][player]
        
        action_values = {}

        # Player's turn
        if node.player == player:
            information_set = node.information_set
            best_response[player][information_set] = {}
            
            nodes_in_information_set = self.get_nodes_in_infoset(information_set, player)
        
            for action in node.actions():
                # Calculate the value of each action 
                action_values[action] = 0
                # Set probability to 0
                best_response[player][information_set][action] = 0
                for possible_node in nodes_in_information_set:
                    # For each possible node (history) the action value is (probability of reaching node * utility of action)
                    action_val = \
                        self.calculate_reach_probability(strategies, possible_node.history) *\
                        self._find_best_response(strategies, possible_node.children[action], best_response, player, information_set_values)
                    # For action value sum over every possible node
                    action_values[action] += action_val
            # Best response is then the action with max action value
            best_response[player][information_set][max(action_values, key=action_values.get)] = 1
            return max(action_values.values())
        
        # Not player's turn
        else:
            for action in node.actions():
                action_values[action] = \
                    strategies[node.player][node.information_set][action] * \
                    self._find_best_response(strategies, node.children[action], best_response, player, information_set_values)

            # return (utility of action * probability of action) for each possible action
            return sum(action_values.values())
    
    def calculate_deltas(self, strategies):
        # Calculate the deltas for each player between best response utility and game values
        game_values = self.calculate_player_values(strategies)
        
        best_response_utility = {}
        for p in self.players:
            # Calculate br(pi_{-i}) = best response to -i
            without_p = {key: val for key, val in strategies.items() if key != p}
            br_p = self.best_response(without_p)
            without_p[p] = br_p[p]
            
            # Calculate u_i using br with fixed -i
            best_response_utility[p] = self.calculate_player_values(without_p)[p]
        
        # delta_i = u_i(br(pi_-i), pi_-i) - u_i(pi)
        # delta_i = (utility of the best response to fixed -i) - (actual utility)
        deltas = {p: best_response_utility[p] - game_values[p] for p in self.players}
        return deltas
    
    def calculate_nash_conv(self, strategies) -> float:
        # Calculate the Nash convergence for a given set of strategies
        # Sum of deltas
        return sum(self.calculate_deltas(strategies).values())
    
    def calculate_exploitability(self, strategies) -> float:
        # Calculate the exploitability for a given set of strategies
        # Sum of deltas / N
        return self.calculate_nash_conv(strategies) / len(self.players)
    
    def uniform_strat_for_player(self, player):
        # Create a uniform strategy for a given player
        strat = {player: {}}
        all_infosets = self.get_infosets()
        for infoset, nodes in all_infosets.items():
            # Find the infosets where it is player's turn
            nodes = [n for n in nodes if n.player == player]
            if len(nodes) == 0:
                continue
            
            # Get actions
            node = nodes[0]
            actions = node.actions()
            
            # Probability of playing a is 1 / |A|
            count = len(actions)
            strat[player][infoset] = {a: 1 / count for a in actions}
        return strat
    
    def self_play(self, chance_strategies=None, iterations=50):
        # Initialize chance strategies for all chance players using a uniform strategy
        chance_strategies = chance_strategies or {c: self.uniform_strat_for_player(c)[c] for c in self.chance}
        
        # Initialize player strategies for both players with a uniform strategy
        p1_strategies = [self.uniform_strat_for_player(self.players[0])]
        p2_strategies = [self.uniform_strat_for_player(self.players[1])]
        
        # List to store exploitabilities at each iteration
        exploitabilities = []
        
        # Run the self-play iterations
        for i in range(iterations):
            # Update player 1's strategy with the latest one and incorporate chance strategies
            p1_strat = p1_strategies[-1]
            p1_strat.update(chance_strategies)
            
            # Update player 2's strategy with the latest one and incorporate chance strategies
            p2_strat = p2_strategies[-1]
            p2_strat.update(chance_strategies)
            
            # Find the best response strategy for player 1 against player 2's strategy
            br_to_p1 = self.best_response(p1_strat)
            br_to_p1.update(chance_strategies)
            
            # Find the best response strategy for player 2 against player 1's strategy
            br_to_p2 = self.best_response(p2_strat)
            br_to_p2.update(chance_strategies)
            
            # Append the new strategies to the lists
            p1_strategies.append(br_to_p2)
            p2_strategies.append(br_to_p1)
            
            # Calculate the average strategies over all iterations for both players and chance players
            strategies = {
                self.players[0]: self.average_strategy(p1_strategies)[self.players[0]],
                self.players[1]: self.average_strategy(p2_strategies)[self.players[1]],
            }
            strategies.update(chance_strategies)
            
            # Calculate and store the exploitability at each iteration
            exploitabilities.append(self.calculate_exploitability(strategies))
            
        return p1_strategies, p2_strategies, exploitabilities

    def regret_matching_cfr(self, regrets, player):
        # Calculate strategy using regret matching for a given player
        strat = {
            player: {}
        }
        for infoset in self.infosets:
            nodes = self.get_nodes_in_infoset(infoset, player)
            if len(nodes) == 0:
                continue

            strat[player][infoset] = {}
            # Get R+ (non-negative regret)
            regrets_plus = {k: max(v, 0) for k,v in regrets[infoset].items()}
            reg_sum = sum(regrets_plus.values())
            for a in self.actions_in_infoset[infoset]:
                # If atleast one action has positive regret
                if reg_sum > 0:
                    strat[player][infoset][a] = regrets_plus[a] / reg_sum
                # Else uniform
                else:
                    strat[player][infoset][a] = 1 / len(self.actions_in_infoset[infoset])
        return strat

    def _prepare_regrets(self, player):
        # Initialize regrets for a given player for all information sets and actions
        regrets = {}
        for infoset in self.infosets:
            nodes = self.get_nodes_in_infoset(infoset, player)
            if len(nodes) == 0:
                continue
            regrets[infoset] = {}
            for a in self.actions_in_infoset[infoset]:
                regrets[infoset][a] = 0
        return regrets

    def probability_of_reaching_state(self, strategies, h1, h2):
        # Calculate the probability of reaching history h2 starting from history h1 using given strategies
        node = self.get_node_by_history(h1)
        
        if not h2.startswith(h1):
            # Cannot be reached
            return 0
        
        # Get path from h1 to h2
        history = h2[len(h1):]
        prob = 1
        
        for a in history:
            if node.player in strategies:
                prob *= strategies[node.player][node.information_set][a]
            node = node.children[a]
        return prob

    
    def utility_in_infoset(self, strategies, infoset, player):
        # Calculate the utility in a specific information set for a given player using given strategies
        infoset_reach_prob = self.calculate_reach_probability_infoset(strategies, infoset)
        if infoset_reach_prob == 0:
            # Infoset cannot be reached
            return 0
        without_p = {key: val for key, val in strategies.items() if key != player}
        
        utility = 0
        terminal_nodes = [self.get_node_by_history(h) for h in self.matrix.keys()]
        # For every node h1 in the information set
        for node in self.get_nodes_in_infoset(infoset, player):
            # For every terminal node h2 in the tree
            for terminal_node in terminal_nodes:
                # Utility is:
                # the probability of reaching infoset (if the player is trying to reach it)
                # multiplied by the probability of reaching h2 from h1
                # multiplied by the reward in h2
                # for every h1, h2
                utility += \
                    self.calculate_reach_probability(without_p, node.history) * \
                    self.probability_of_reaching_state(strategies, node.history, terminal_node.history) * \
                    self.matrix[terminal_node.history][player]
                    
        # Normalize by the probability of reaching the infoset
        utility /= infoset_reach_prob
        return utility

    def update_regrets(self, regrets, strategies, player):
        # pi_-i
        without_p = {key: val for key, val in strategies.items() if key != player}

        for infoset in self.get_infosets():
            nodes = self.get_nodes_in_infoset(infoset, player)
            if len(nodes) == 0:
                continue
            
            for a in self.actions_in_infoset[infoset]:
                # Regret in information set I and action a = R(I,a)
                # is calculated against a strategy that is identical to the player's strategy 
                # but in infoset I the probability of playing action a is 1
                strategies_where_a_in_infoset = copy.deepcopy(strategies)
                for a2 in self.actions_in_infoset[infoset]:
                    strategies_where_a_in_infoset[player][infoset][a2] = 0
                strategies_where_a_in_infoset[player][infoset][a] = 1
                
                # R(I,a) = 
                # probability of reaching I if the player is trying to reach it *
                # (utility with strategies but the player plays a in I - utility with strategies)
                regrets[infoset][a] += \
                    self.calculate_reach_probability_infoset(without_p, infoset) * \
                    ( \
                        self.utility_in_infoset(strategies_where_a_in_infoset, infoset, player) - \
                        self.utility_in_infoset(strategies, infoset, player) \
                    )
                

    def CFR(self, chance_strategies=None, iterations=50):
        chance_strategies = chance_strategies or {c: self.uniform_strat_for_player(c)[c] for c in self.chance}
        
        # Dict to store the regrets 
        regrets = {p: self._prepare_regrets(p) for p in self.players}
        
        # Lists to store used strategies 
        player_strategies = {p: [] for p in self.players}
        exploitabilities = []
        
        for _ in range(iterations):
            # Get new strat T from regrets until T-1 
            for p in self.players:
                player_strategies[p].append(self.regret_matching_cfr(regrets[p], p))
            
            # Calculate exploitability from average strategies
            current_strategies = {}
            for p in self.players:
                current_strategies.update(self.average_strategy(player_strategies[p]))
            current_strategies.update(chance_strategies)
            
            exploitabilities.append(self.calculate_exploitability(current_strategies))
            
            # Update regrets using current strategies
            for p in self.players:
                self.update_regrets(regrets[p], current_strategies, p)
        
        return player_strategies, exploitabilities


def plot_exploitability(exploitabilities: np.array):
    plt.plot(list(range(len(exploitabilities))), exploitabilities)
    plt.show()


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

def kuhn_poker() -> ExtensiveFormGame:
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
                    }, "KAB")
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
                    }, "AAB")
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


def pretty_print(strat, ind=''):
    if isinstance(strat, dict):
        for k in strat.keys():
            print(f'{ind}{k}: ')#'+'{')
            pretty_print(strat[k], ind=ind+'\t')
            # print(f'{ind}'+'}')
    else:
        print(f'{ind}{strat}')
        
def pretty_print_strat(strat: dict):
    for player, strategy in strat.items():
        print(f"Player: {player}")
        for infoset, probabilities in strategy.items():
            print(f"  information set: {infoset}")
            for action, probability in probabilities.items():
                print(f"    action: {action}, probability: {probability:.4f}")
        print()


def example_cfr(tree: ExtensiveFormGame, chance_strategies: dict, iterations: int=50):
    strats, expl = tree.CFR(iterations=iterations, chance_strategies=chance_strategies)
    strats = {p: tree.average_strategy(s) for p,s in strats.items()}
    return strats, expl

def example_sp(tree: ExtensiveFormGame, iterations: int=50):
    p1, p2, expl = tree.self_play(iterations=iterations)
    return tree.average_strategy(p1), tree.average_strategy(p2), expl

def summary_alg(name: str, strategies: dict, exploitability: float):
    print(f"{name}:")
    pretty_print_strat(strategies)
    print(f"exploitability {exploitability:.4f}")


def cfr_vs_selfplay(tree: ExtensiveFormGame, chance_strategies: dict=None, iterations: int=50):
    chance_strategies = chance_strategies or {c: tree.uniform_strat_for_player(c)[c] for c in tree.chance}
    cfr_strats, expl_cfr = example_cfr(tree, chance_strategies=chance_strategies, iterations=iterations)
    p1_cfr, p2_cfr = cfr_strats["Player1"], cfr_strats["Player2"]
    p1_sp, p2_sp, expl_sp = example_sp(tree, iterations=iterations)
    
    print(f"iterations: {iterations}")
    strategies_sp = copy.deepcopy(p1_sp)
    strategies_sp.update(p2_sp)
    strategies_cfr = copy.deepcopy(p1_cfr)
    strategies_cfr.update(p2_cfr)
    summary_alg("SELFPLAY", strategies_sp, expl_sp[-1])
    summary_alg("CFR", strategies_cfr, expl_cfr[-1])
    print()
    
    print(f"p1 SELFPLAY vs p2 CFR:")
    strategies = copy.deepcopy(p1_sp)
    strategies.update(p2_cfr)
    strategies.update(chance_strategies)
    game_val = tree.calculate_player_values(strategies)
    print(f"p1 {game_val[tree.players[0]]}, p2 {game_val[tree.players[1]]}")
    print()
    
    print(f"p1 CFR vs p2 SELFPLAY:")
    strategies = copy.deepcopy(p1_cfr)
    strategies.update(p2_sp)
    strategies.update(chance_strategies)
    game_val = tree.calculate_player_values(strategies)
    print(f"p1 {game_val[tree.players[0]]}, p2 {game_val[tree.players[1]]}")
    

if __name__=="__main__":
    # tree = rps()
    tree = kuhn_poker()
    
    cfr_vs_selfplay(tree, iterations=500)


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


# tree = avg_example()
# list_strat = [
#     {"Player1":
#         {
#             "": {"A": 0.2, "B": 0.8},
#             "B2": {"A": 0.8, "B": 0.2}
#         }
#     },
#     {"Player1":
#         {
#             "": {"A": 0.8, "B": 0.2},
#             "B2": {"A": 0.2, "B": 0.8}
#         }
#     },
# ]

# print("Strat1")
# print(list_strat[0])
# print("Strat2")
# print(list_strat[1])
# avg_strat = tree.average_strategy(list_strat)
# print("avg strat")
# print(avg_strat)
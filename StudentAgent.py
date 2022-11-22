import torch
import math
import numpy as np
from treelib import Node, Tree
import itertools
import board as B
from random import randrange
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class State(object):
    '''
    Class to store the representation of the game at each
    node of the tree.
    Contains the gameboard, the position of the piece that just moved
    Tag is a string representation of the position
    '''
    def __init__(self, position, removed):
        if type(position) == tuple:
            position = np.array([position[0], position[1]])
        self.position = position
        self.removed = removed
        self.tag = np.array2string(position)
        if removed is None:
            self.remove_tag = 'None'
        else:
            self.remove_tag = np.array2string(removed)

class Student_Agent(object):
    def __init__(self, side:str, board=None):
        self.side = side
        self.board = None

    def get_valid_directions(self, start_pos, end_pos, promoted):
        valid_directions = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]])

        if promoted:
            return valid_directions
        else:
            if start_pos is not None:
                last_direction = self.get_last_direction(start_pos, end_pos)
                idx = np.where((valid_directions == last_direction).all(axis=1))
                valid_directions = np.delete(valid_directions, idx, axis=0)

            if self.side == "White":
                backwards = np.array([1, 0])
            elif self.side == "Black":
                backwards = np.array([-1, 0])

            idx = np.where((valid_directions == backwards).all(axis=1))
            valid_directions = np.delete(valid_directions, idx, axis=0)
            
            return valid_directions
    
    def get_last_direction(self, start_pos, end_pos):
        direction = start_pos - end_pos
        
        for i, element in enumerate(direction):
            if element > 0:
                direction[i] = 1
            elif element < 0:
                direction[i] = -1

        return direction

    def check_for_promotions(self, temp_gameboard=None):
        if temp_gameboard is None:
            gameboard = self.board
        else:
            gameboard = temp_gameboard

        for i in [0, 7]:
            for j in range(8):
                pos = (i, j)
                if i == 7:
                    if gameboard.loc(pos) == 1:
                        gameboard.board[i][j] = 2
                elif i == 0:
                    if gameboard.loc(pos) == 3:
                        gameboard.board[i][j] = 4
        
    def get_all_legal_moves(self, side, temp_gameboard=None):
        """
        Input: 
            side: string, "White" or "Black"
            temp_gameboard: Optional, if None, use self.board as current gammeboard
        
        Output:
            move: a list of legal move paths, a legal move path is a list including the index of every step in a move path, starting from the start position to the end position
            remove: a list of captures for each legal move path, remove[i] is the list of captured piece index for the move path i
            count: a list of the number of captured pieces for each legal move path, count[i] is the number of captured pieces for the move path i
            board: a list of output board after executing each legal move path, board[i] is the output board after performing the move path i
        """

        if temp_gameboard is None:
            gameboard = self.board
        else:
            gameboard = B.Board(board=np.copy(temp_gameboard.board))

        all_move_list = []
        all_remove_list = []
        all_count_list = []

        for pos in itertools.product(list(range(8)), repeat=2):
            if gameboard.player_owns_piece(side, pos):
                # time1 = time.time()
                all_possible_move_tree = self.get_piece_legal_move(side, pos, current_gameboard=gameboard)
                # time2 = time.time()

                if all_possible_move_tree.depth() > 0:
                    b = self.listFromTree(all_possible_move_tree)

                    all_move_list.extend(b['move'])
                    all_remove_list.extend(b['remove'])
                    all_count_list.extend(b['count'])

        if len(all_count_list) > 0:
            max_indices = np.argwhere(all_count_list == np.amax(all_count_list)).flatten().tolist()

            valid_moves = [all_move_list[i] for i in max_indices]
            valid_removes = [all_remove_list[i] for i in max_indices]
            valid_counts = [all_count_list[i] for i in max_indices]
            #print("b[move]", b['move'], b['remove'])
            #valid_board = [self.performMove(b['move'][0], b['remove'][0], deepcopy(gameboard)) for i in max_indices]
            valid_board = [self.performMove(all_move_list[i], all_remove_list[i], deepcopy(gameboard)) for i in max_indices]

            return {
                'move' : valid_moves,
                'remove' : valid_removes,
                'count' : valid_counts,
                'board' : valid_board,
            }
        else:
            return {
                'move' : [],
                'remove' : [],
                'count' : [],
                'board' : [],
            }


    def get_piece_legal_move(
        self, player, position, startPosition=None, current_gameboard=None, lastRemoved=None,
        movetree=None, lastNode=None, canMove=True, hasJumped = False
    ):

        '''
        position is the current position of the piece whose moves we are inspecting
        startPosition is the original position of that move, before any jumps have been made
        '''
        # Initialize empty lists
        if current_gameboard is None:
            current_gameboard = self.board

        # Check for promotions
        self.check_for_promotions(current_gameboard)

        # Add the node to the movetree, or create one if it doesn't exist
        if movetree is None:
            movetree = Tree()

        if current_gameboard.player_owns_piece(self.side, position):
            
            # Create a node for the tree from the current state of the game
            state = State(position, lastRemoved)
            node = Node(tag=state.tag, data=state)

            # if current_gameboard.player_owns_piece(player, position):
            if lastNode is None:
                # Set current node as the root
                movetree.add_node(node)
                lastNode = node
            else:
                # Create a new node as the child of the last node
                movetree.add_node(node, parent=lastNode)
            
            valid_directions = self.get_valid_directions(
                startPosition, position, current_gameboard.is_promoted(position)
            )

            if current_gameboard.is_promoted(position):
                lookup_range = 8
            else:
                lookup_range = 3

            for direction in valid_directions:
                
                jumpIsAvailable = False
                jumpablePiece = None

                for multiplier in range(1, lookup_range):

                    if not current_gameboard.is_promoted(position):
                        if multiplier == 2 or hasJumped:
                            canMove = False
                        elif multiplier == 1 and not hasJumped:
                            canMove = True
                    
                    next = position + multiplier * direction
                    next_next = position + (multiplier + 1) * direction

                    # Check for any collision or invalid moves

                    # Out of board
                    # Quit
                    if current_gameboard.is_outside_board(next):
                        break

                    # You own the next piece
                    # Quit
                    elif current_gameboard.player_owns_piece(self.side, next):
                        break
                    
                    # Collion with two back to back pieces
                    # Quit
                    elif (
                        not current_gameboard.loc(next) == 0
                        and not current_gameboard.is_outside_board(next_next)
                        and not current_gameboard.loc(next_next) == 0
                        and not current_gameboard.player_owns_piece(self.side, next)
                        and not current_gameboard.player_owns_piece(self.side, next_next)
                    ):
                        break
                    
                    if current_gameboard.loc(next) == 0:
                        if jumpIsAvailable:
                            if current_gameboard.opponents_between_two_positions(self.side, position, next) < 2:
                                temp_gameboard = B.Board(board=np.copy(current_gameboard.board))
                                temp_gameboard.move_piece(position, next)
                                temp_gameboard.remove_piece(jumpablePiece)
                                # print(">>>>>>>>")
                                # temp_gameboard.visualize_board()

                                self.get_piece_legal_move(
                                    self.side, next, startPosition = position,
                                    current_gameboard = temp_gameboard, lastRemoved=jumpablePiece,
                                    movetree = movetree, lastNode = node, canMove = False, hasJumped = True
                                )

                        elif canMove:
                            temp_gameboard = B.Board(board=np.copy(current_gameboard.board))
                            temp_gameboard.move_piece(position, next)

                            new_state = State(next, None)
                            new_node = Node(tag=new_state.tag, data=new_state)

                            movetree.add_node(new_node, parent=node)
                    elif (
                        not current_gameboard.loc(next) == 0
                        and not current_gameboard.player_owns_piece(self.side, next)
                    ):
                        if not jumpIsAvailable:
                            jumpIsAvailable = True
                            jumpablePiece = next

        return movetree
    
    def nextMove(self, board):
        """
        Implement this method by your own
        Input: the current gameboard
        Output: the index of the move path choice, the index should be within list(range(len(self.get_all_legal_moves(self.side, board)['move'])))
        """
        choices = self.get_all_legal_moves(self.side, board)
        choice = randrange(len(choices['move']))
        #print("getting random choice", choice)
        return choice

    def performMove(self, moveList, removeList, temp_gameboard=None):

        if temp_gameboard is None:
            gameboard = self.board
        else:
            # gameboard = B.Board(board=np.copy(self.board.board))
            gameboard = temp_gameboard

        for i in range(len(moveList)):
            if i == 0:
                pass
            else:
                gameboard.move_piece(moveList[i-1], moveList[i])
                if removeList[i] is not None:
                    gameboard.remove_piece(removeList[i])
        
        return deepcopy(gameboard.board)

    def listFromTree(self, tree):
        tag_paths = []
        remove_paths = []
        count_list = []
        for i in tree.paths_to_leaves():
            path = []
            r = []
            for j in i:
                path.append(tree.get_node(j).data.position)
                r.append(tree.get_node(j).data.removed)

            tag_paths.append(path)
            remove_paths.append(r)
            count_list.append(self.countRemoves(r))
        
        return {
            'move' : tag_paths,
            'remove' : remove_paths,
            'count' : count_list
        }

    def setFromTree(self, tree):
        tag_paths = []
        remove_paths = []
        for i in tree.paths_to_leaves():
            path = []
            r = []
            for j in i:
                path.append(tree.get_node(j).data.tag)
                r.append(tree.get_node(j).data.remove_tag)

            tag_paths.append(path)
            remove_paths.append(r)
        
        move_set = set(map(tuple, tag_paths))
        remove_set = set(map(tuple, remove_paths))

        return {
            'move' : move_set,
            'remove' : remove_set
        }

    def countRemoves(self, remove_list):
        count = 0
        for i in remove_list:
            if i is not None:
                count = count + 1
        return count






class NodeMCTS:
    def __init__(self, prior, to_play, board):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.board = board

    
    
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    #         def select_action(self, temperature):
    #             """
    #             Select action according to the visit count distribution and the temperature.
    #             """
    #             visit_counts = np.array([child.visit_count for child in self.children.values()])
    #             actions = [action for action in self.children.keys()]
    #             if temperature == 0:
    #                 action = actions[np.argmax(visit_counts)]
    #             elif temperature == float("inf"):
    #                 action = np.random.choice(actions)
    #             else:
    #                 # See paper appendix Data Generation
    #                 visit_count_distribution = visit_counts ** (1 / temperature)
    #                 visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
    #                 action = np.random.choice(actions, p=visit_count_distribution)

    #             return action

    def select_child(self):
        def ucb_score(parent, child):
            """
            The score for an action that would transition between the parent and child.
            """
            prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
            if child.visit_count > 0:
                # The value of the child is from the perspective of the opposing player
                value_score = -child.value()
            else:
                value_score = 0

            return value_score + prior_score
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = []
        best_child = None

        #for action, child in self.children.items():
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
       #print(best_child)

       # return best_action, best_child
        return  best_action, best_child

    def expand(self, student_player, to_play, model):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        best = -1
        self.to_play = to_play
        #print(to_play)

        board = deepcopy(self.board)
        #print(board.board)
        student_player.side = to_play
        configurations = student_player.get_all_legal_moves(to_play, board)
        if self.to_play == "White":
            to_play = "Black"
        else:
            to_play = "White"
            best = 100
        
        #print("configs: ", configurations, " ")
        move = []
        remove = []
        for i in range(len(configurations['move'])):
            board = deepcopy(self.board)
            #print(configurations['board'])
            prob = model.predict(configurations['board'][i])
           # print(prob)
            board.board = student_player.performMove(configurations['move'][i], configurations['remove'][i], deepcopy(self.board))
            board.increment_turn()
            temp = ((configurations['move'][i][0][0], configurations['move'][i][0][1]), (configurations['move'][i][1][0],configurations['move'][i][1][1]))    
            
            if self.to_play == "White" and prob[0] > best:
            #    print("White has moved: ", board.board)
                best = prob[0]  
                self.children[temp] = NodeMCTS(prior=float(prob[0]), to_play=to_play,board=board)
            elif self.to_play == "Black" and prob[0] < best:
            #    print("Black has moved: ", board.board)
                best = prob[0]
                self.children[temp] = NodeMCTS(prior=float(prob[0]), to_play=to_play,board=board)
         




    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.board.__str__(), prior, self.visit_count, self.value())

class MCTS:

    def __init__(self, board, move_checker, student_player, model, num_simulations):
        self.student_player = student_player
        self.move_checker = move_checker
        self.board = board
        self.model = model
        self.num_simulations = num_simulations


    
    
    def run(self, to_play):

        #Created root node
        #optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        root = NodeMCTS(0, to_play, deepcopy(self.board))               

        # EXPAND root
        #value = model.predict(board)
        #valid_moves = self.move_checker.get_all_legal_moves(board)
#                 action_probs = action_probs * valid_moves  # mask invalid moves
#                 action_probs /= np.sum(action_probs)

        #Expand the root node
        root.expand( self.student_player, to_play, self.model)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                #print(node.board.board)
                action, node = node.select_child()
                search_path.append(node)
           # print(node.board.board)
            
           # print("Search path:          ", search_path, "       End path")
            parent = search_path[-2]
            board = parent.board
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective


            # The value of the new state from the perspective of the other player

            win = board.check_win(parent.to_play, action)
            if win is None:
                # If the game has not ended:
                # EXPAND
                node.expand( self.student_player, node.to_play, self.model)
                action, node = node.select_child()
                #print(node.board.board)
                search_path.append(node)
                value = board.check_win(node.to_play, action)
                
            to_play = node.to_play
            temp_board = deepcopy(node.board)
            while win is None:
               # print(temp_board.board)
                temp_board.increment_turn()
                self.move_checker.side = to_play
                self.student_player.side = to_play
                moves = self.move_checker.get_all_legal_moves(deepcopy(temp_board))
               # print(moves)
                win = temp_board.check_win(to_play, moves['move'])                  
                if win is not None:
                    if to_play == "White":
                        to_play = "Black"
                    else:
                        to_play = "White"                                              
                    break
           
                choice_idx = self.student_player.nextMove(deepcopy(temp_board))
                temp_board.board = self.student_player.performMove(moves['move'][choice_idx], moves['remove'][choice_idx], deepcopy(temp_board))
                if to_play == "White":
                    to_play = "Black"
                else:
                    to_play = "White"   

                                                                                                         
            
            value = 1

            self.backpropagate(search_path, value, to_play)                   

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        print(search_path)
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


class NN(nn.Module):

    def __init__(self):

        super(NN, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size = 64

        self.fc1 = nn.Linear(in_features=self.size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))


        return x

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            x = self.forward(board)

        return x

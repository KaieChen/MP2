import numpy as np
from treelib import Node, Tree
import itertools
import board as B
from random import randrange
from copy import deepcopy

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

class Ta_Agent(object):
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
                pos = [i, j]
                if i == 7:
                    if gameboard.loc(pos) == 1:
                        gameboard.board[i][j] = 2
                elif i == 0:
                    if gameboard.loc(pos) == 3:
                        gameboard.board[i][j] = 4
        
    def get_all_legal_moves(self, temp_gameboard=None):

        if temp_gameboard is None:
            gameboard = self.board
        else:
            # gameboard = B.Board(board=np.copy(temp_gameboard.board))
            gameboard = temp_gameboard

        all_move_list = []
        all_remove_list = []
        all_count_list = []

        for pos in itertools.product(list(range(8)), repeat=2):
            if gameboard.player_owns_piece(self.side, pos):
                # time1 = time.time()
                all_possible_move_tree = self.get_piece_legal_move(self.side, pos, current_gameboard=gameboard)
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

            return {
                'move' : valid_moves,
                'remove' : valid_removes,
                'count' : valid_counts
            }
        else:
            return {
                'move' : [],
                'remove' : [],
                'count' : []
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

    def nextMove(self, candidate_moves):
        choice = randrange(len(candidate_moves['move']))
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
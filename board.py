import sys
import numpy as np
import math
from copy import deepcopy
from treelib import Node, Tree
import itertools
import TaAgent
import StudentAgent
import time

"""
EMPTY = 0
BLACK_MAN = 1
BLACK_KING = 2
WHITE_MAN = 3
WHITE_KING = 4
"""

class Board(object):
    def __init__(self, board=None):
        if board is not None:
            self.board = board
        else:
            self.board = np.zeros((8, 8), dtype=np.int32)
            self.board[1:3, :] = 1 # set all black men
            self.board[5:7, :] = 3 # set all white men
        self.white_piece_symbol = "X"
        self.black_piece_symbol = "Y"
        self.white_pieces = [3, 4]
        self.black_pieces = [1, 2]
        self.white_man = 3
        self.white_king = 4
        self.black_man = 1
        self.black_king = 2
        self.n_turns = 0  
    
    def loc(self, pos):
        if not self.is_outside_board(pos):
            return self.board[pos[0], pos[1]]
        else:
            raise IndexError("Position outside of board")

    def player_owns_piece(self, player, pos):
        if not self.is_outside_board(pos):
            if self.board[pos[0], pos[1]] in self.white_pieces and player == "White":
                return True
            elif self.board[pos[0], pos[1]] in self.black_pieces and player == "Black":
                return True
            else:
                return False
        else:
            raise IndexError("Position outside of board")
    
    def is_outside_board(self, pos):
        if pos[0] in list(range(8)) and pos[1] in list(range(8)):
            return False
        else:
            return True
    
    def visualize_board(self):
        str_board = self.board.tolist()
        for board_row in str_board:
            board_row = [str(num) for num in board_row]
            print(" ".join(board_row))

    def is_promoted(self, pos: tuple):
        if ((self.board[pos[0], pos[1]] == 2) or 
            (self.board[pos[0], pos[1]] == 4)):
            return True
        else:
            return False

    def check_king_promotion(self):
        white_king_row = self.board[0, :]
        white_king_row[white_king_row == 3] = 4
        black_king_row = self.board[7, :]
        black_king_row[black_king_row == 1] = 2
        self.board[0, :] = white_king_row[:]
        self.board[7, :] = black_king_row[:]

    def measure(self, player):
        if player == "Black":
            myPieces = np.count_nonzero(self.board == 1)
            myPromoted = np.count_nonzero(self.board == 2)
            opponentPieces = np.count_nonzero(self.board == 3)
            opponentPromoted = np.count_nonzero(self.board == 4)
        else:
            myPieces = np.count_nonzero(self.board == 3)
            myPromoted = np.count_nonzero(self.board == 4)
            opponentPieces = np.count_nonzero(self.board == 1)
            opponentPromoted = np.count_nonzero(self.board == 2)

        return {
            'myPieces' : myPieces,
            'myPromoted' : myPromoted,
            'opponentPieces' : opponentPieces,
            'opponentPromoted' : opponentPromoted
        }

    def check_win(self, player, moves):
        metric = self.measure(player)
        if (
            # No moves left
            len(moves) == 0 or
            (
            # One man versus One King
            metric['myPieces'] == 1
            and metric['myPromoted'] == 0
            and metric['opponentPieces'] == 0
            and metric['opponentPromoted'] == 1
            )
        ):
            return True 
    
    def remove_piece(self, pos):
        self.board[pos[0], pos[1]] = 0
      
    def move_piece(self, start_pos, end_pos):
        piece = self.loc(start_pos)
        self.remove_piece(start_pos)
        self.board[end_pos[0], end_pos[1]] = piece

    def opponents_between_two_positions(self, player, p1, p2):
        if p1[0] == p2[0]:
            x1 = p1[0]
            y1 = min(p1[1], p2[1])
            y2 = max(p1[1], p2[1])
            res = self.board[x1, :][y1:y2+1]
        elif p1[1] == p2[1]:
            y1 = p2[1]
            x1 = min(p1[0], p2[0])
            x2 = max(p1[0], p2[0])
            res = self.board[:, y1][x1:x2+1]

        count = 0
        for i in res:
            if player == "White":
                if i == 1 or i == 2:
                    count = count + 1
            elif player == "Black":
                if i == 3 or i == 4:
                    count = count + 1
        
        return count

    def increment_turn(self):
        self.n_turns += 1

class Student_Move_Checker(object):
    def __init__(self, side, board=None):
        self.board = Board(board=board)
        self.side = side
    
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
        
    def get_all_legal_moves(self, temp_gameboard=None):

        if temp_gameboard is None:
            gameboard = self.board
        else:
            gameboard = Board(board=np.copy(temp_gameboard.board))

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
            state = TaAgent.State(position, lastRemoved)
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
                                temp_gameboard = Board(board=np.copy(current_gameboard.board))
                                temp_gameboard.move_piece(position, next)
                                temp_gameboard.remove_piece(jumpablePiece)

                                self.get_piece_legal_move(
                                    self.side, next, startPosition = position,
                                    current_gameboard = temp_gameboard, lastRemoved=jumpablePiece,
                                    movetree = movetree, lastNode = node, canMove = False, hasJumped = True
                                )

                        elif canMove:
                            # gameboard gameboard gameboard
                            temp_gameboard = Board(board=np.copy(current_gameboard.board))
                            temp_gameboard.move_piece(position, next)

                            new_state = TaAgent.State(next, None)
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

    def check_next_move(self, board, start_pos, end_pos):
        self.board = board
        candidate_moves = self.get_all_legal_moves()
        possible_pos_list = self.possible_pos(candidate_moves)
        for choice_idx, start_end_pos in enumerate(possible_pos_list):
            if (start_end_pos[0][0] == start_pos[0] 
                and start_end_pos[0][1] == start_pos[1] 
                and start_end_pos[1][0] == end_pos[0] 
                and start_end_pos[1][1] == end_pos[1]):
                return choice_idx
        return -1
    
    def possible_pos(self, moves):
        possible_pos_list = []
        for possible_move in moves['move']:
            start_pos = [possible_move[0][0], possible_move[0][1]]
            end_pos = [possible_move[-1][0], possible_move[-1][1]]
            possible_pos_list.append([start_pos, end_pos])
        return possible_pos_list
    
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

if __name__=="__main__":

    win_count = {"Student": 0, "TA": 0}

    for game in range(5):
        user_board = Board()
        move_checker = Student_Move_Checker("White")
        student_player = StudentAgent.Student_Agent("White")
        ta_player = TaAgent.Ta_Agent("Black")
        
        game_playing = True

        while game_playing:
            user_board.increment_turn()

            moves = move_checker.get_all_legal_moves(deepcopy(user_board))
            if user_board.check_win(student_player.side, moves['move']):
                win_count["TA"] += 1
                break
            start_time = time.time()
            choice_idx = student_player.nextMove(deepcopy(user_board))
            assert (time.time() - start_time) < 1.01
            
            user_board.board = student_player.performMove(moves['move'][choice_idx], moves['remove'][choice_idx], deepcopy(user_board))

            ta_moves = ta_player.get_all_legal_moves(deepcopy(user_board))
            if user_board.check_win(ta_player.side, ta_moves['move']):
                win_count["Student"] += 1
                break
            ta_choice_idx = ta_player.nextMove(ta_moves)
            user_board.board = ta_player.performMove(ta_moves['move'][ta_choice_idx], ta_moves['remove'][ta_choice_idx], deepcopy(user_board))

            if user_board.n_turns >= 500:
                win_count["Student"] += 1
                game_playing = False

    for game in range(5):
        user_board = Board()
        move_checker = Student_Move_Checker("Black")
        student_player = StudentAgent.Student_Agent("Black")
        ta_player = TaAgent.Ta_Agent("White")
        
        game_playing = True

        while game_playing:
            user_board.increment_turn()

            ta_moves = ta_player.get_all_legal_moves(deepcopy(user_board))
            if user_board.check_win(ta_player.side, ta_moves['move']):
                win_count["Student"] += 1
                break
            ta_choice_idx = ta_player.nextMove(ta_moves)
            user_board.board = ta_player.performMove(ta_moves['move'][ta_choice_idx], ta_moves['remove'][ta_choice_idx], deepcopy(user_board))
            
            moves = move_checker.get_all_legal_moves(deepcopy(user_board))
            if user_board.check_win(student_player.side, moves['move']):
                win_count["TA"] += 1
                break
            start_time = time.time()
            choice_idx = student_player.nextMove(deepcopy(user_board))
            assert (time.time() - start_time) < 1.01
            
            user_board.board = student_player.performMove(moves['move'][choice_idx], moves['remove'][choice_idx], deepcopy(user_board))

            if user_board.n_turns >= 500:
                win_count["Student"] += 1
                game_playing = False
    
    print(win_count)


# WSL       : Ubuntu-20.04
# Python    : 3.8.10 64-bit

from math import inf        # 탐사되지 않은 child의 UTC값은 inf로 표현된다.
from math import sqrt       # UTC를 구하는데 사용
from math import log as ln  # UTC를 구하는데 사용
from random import randint  # 같은 최고값의 potential(UTC)을 갖는 child가 많을때
                            # random 선택을 위해 사용
from copy import deepcopy   # 제귀 과정에서 board를 전수하는데 에러를 막기 위해 사용
from time import time       # 학습시 시간적 제약을 주기 위해 사용


class mcts_node:
    # root node 만들때
    # mcts_node(None, 0, None, board, None, 0, in_number_of_players)
    # child node 만들때
    # mcts_node(parent, parent.depth + 1, move, board, state, turn, len(parent.win))
    def __init__(self, in_parent, in_depth, in_move, in_board, in_state, in_turn, in_number_of_players):
        self.parent = in_parent     # curr state
        self.child = []             # 여러가지 outcome을 대비해 child를 list로 설정
        self.depth = in_depth       # node의 깊이 - 닭 달걀과 같은 순환의 문제를 방지하기 위해
                                    # node들을 merge 시킬때는 같은 depth의 node끼리만
                                    # merge할 수 있도록 한다.
        
        self.move = in_move                 # 이전 노드에서 현제 노드에 오게된 움직임을 저장
        self.board = deepcopy(in_board)     # 현제 게임의 진행상황을 나타냅니다.
        self.state = in_state               # 현제 노드의 상태
                                            # None                 = not explored node
                                            # -2                   = tree node
                                            # -1                   = leaf node 무승부
                                            # winner_player_number = leaf node 승부

        self.turn = in_turn                                     # 현제 노드가 몇번 player의 turn인지
        self.visit = 0                                          # 현제 노드가 몇번 visit 되었는지
        self.win = [0 for idx in range(in_number_of_players)]   # 현제 노드의 하위에서 각 player의 승수가 어떻게 되는지
    
    # 현 시점 가장 유망한 child 의 idx를 return
    # if(self == tree) : return self시점 본인 승률이 가장 높게 나오는 child node의 idx
    # if(self == leaf) : return None
    def Selection(self):
        if(len(self.child) == 0):                               # 들어왔던 node가 leaf node이면
            return None                                         # None를 return
        else:                                                   # 들어왔던 node가 tree node이면
            # 모든 child의 UTC값을 구한다.
            UTCs = []              # child 들의 UTC값을 임시저장할 list
            for child_idx in range(len(self.child)):
                # visit횟수가 0인 child는 potential을 ∞(default)로 설정한다.
                next_state = self.child[child_idx]
                # if(me_visit == 0) : UTC == ∞
                if(next_state.visit == 0):
                    UTCs.append(inf)
                # UTC = me_win / me_visit + 2 * sqrt(ln(parent_visit) / me_visit))
                #       승률 좋은 녀석중에  상수  정확성이 더 낮은녀석부터
                else:
                    win_rate = next_state.win[self.turn] / next_state.visit 
                    Accuracy = sqrt(2) * sqrt(ln(self.visit) / next_state.visit)
                    UTC = win_rate + Accuracy
                    UTCs.append(UTC)

            max_val = max(UTCs)    # potential 최고값
            # potential 최고값과 같은 potential을 가지고 있는 child들의 index를 list에 저장한다.
            max_idx = []
            for child_idx in range(len(self.child)):
                if(max_val == UTCs[child_idx]):
                    max_idx.append(child_idx)

            # 최고값이 여럿 있으면 그중 random한 녀석을 고른다.
            # 가장 potential 이 큰 child 의 index를 보고한다.
            return max_idx[randint(0, len(max_idx) - 1)]


    # if(move[0] == True) : return next in turn player number
    def next_turn(self, move):
        if(move[0] == True):                    # if end turn
            next_turn = self.turn + 1           # pass to next player
            if(next_turn == len(self.win)):     # if next player overflow
                next_turn = 0                   # set next player to player 0
            return next_turn                    # return next player number
        else:                                   # if not end turn
            return self.turn                    # return current player number


    # play 가능한 child들의 outcome을 계산한 뒤 전부 self의 child에 append한다.
    # None                 = tree node 
    # -1                   = leaf node 무승부
    # winner_player_number = leaf node 승부
    def Breadth(self):
        # tree node
        if(self.state == None):                                               
            moves = self.Playable_options()                                     # get all playable options
            for move in moves:                                                  # for every move
                next_board, next_state = self.Play_outcome(move)                # calc outcome
                next_turn = self.next_turn(move)                                # if "pass turn" switch to next player
                new_child = mcts_node(self, self.depth + 1, move, next_board,   # create new child
                                        next_state, next_turn, len(self.win))
                self.child.append(new_child)                                    # append to self
        # leaf node
        elif(self.state == -1):
            print("[ERROR] leaf node cant access Breadth!!")


    # self로부터 시작해서 1회 시뮬레이션, 학습 후 가장 유망한 child의 idx return
    # if(self == tree) : return self시점 본인 승률이 가장 높게 나오는 child node의 idx
    # if(self == leaf) : return None
    def Expansion(self):
        curr_node = self                                        # while 을 통해 leaf 까지 타고 들어갈때 사용
        while(True):                                            # 게임이 끝나지 않는동안
            curr_node.visit += 1                                # 현제 node의 visit 횟수를 올려준다.
            if(curr_node.state == None):                        # tree node 라면
                if(len(curr_node.child) == 0):
                    curr_node.Breadth()                         # 현제 node의 상태를 판별하고 가능한 child 모두 append
                curr_node = curr_node.child[curr_node.Selection()]  # 그 child중 best를 고른다.
            else:                                               # leaf node 라면
                break                                           # break
            
        # Tree의 Root 까지 타고 올라가면서 결과를 update한다.
        curr_node.Backpropagation(curr_node.state)

        # 가장 유망한 child의 idx를 return한다.
        return self.Selection()                             # 가장 유망한 child의 index를 return


    # Tree의 Root 까지 타고 올라가면서 시뮬레이션 결과를 update한다.
    # no return value
    # reward 를 조정해서 학습 속도를 더 높일 수 있을듯.
    def Backpropagation(self, game_result):
        curr_node = self                        # while 을 통해 Root 까지 타고 올라갈때 사용
        while(curr_node != None):
            # 무승부인 경우
            if(game_result == -1):
                for idx in range(len(curr_node.win)):
                    curr_node.win[idx] += 0
            # 승부인 경우
            else:
                for idx in range(len(curr_node.win)):
                    # win
                    if(game_result == idx):
                        curr_node.win[idx] += 1
                    # lose
                    else:
                        curr_node.win[idx] -= 1
            curr_node = curr_node.parent            # 부모는 하나뿐이기 때문에 올라가는것은 간단하다.


    # self로부터 시작해서 반복 시뮬레이션, 학습 후 가장 유망한 child의 idx return
    # if(self == tree) : return self시점 본인 승률이 가장 높게 나오는 child node의 idx
    # if(self == leaf) : return None
    def mcts_learn(self, repetitions, time_limit):
        end_time = time() + time_limit  # time_limit 은 초 단위로 주어져야 한다.
        loop_count = 0                  # 몇회 학습하는지.
        best_child = None
        while (time() < end_time and loop_count < repetitions):
            best_child = self.Expansion()
            loop_count += 1
        return best_child               # 가장 유망한 child를 return


#==============================아래 함수는 게임마다 다르게 함수를 구현해야함==============================

    # move = [True/false, [row, col]]
    #           turn end,

    # return [list of possible moves]
    def Playable_options(self):
        moves = []          # move = [True/false, play_data]
                            #           turn end, play command

        #======================#
        #                      #
        #    게임마다 다르게    #
        #                      #
        #======================#
        # playable한 모든 option을 moves에 list로 담아야함.
        # move = [True/false, play_data] 의 play_data를 바로 game에 적용해도 문제가 없어야함.

        for row in range(3):
            for col in range(3):
                if(self.board[row][col] == '.'):
                    moves.append([True, [row, col]])

        return moves
    

    # return [next_board, next_state]
    def Play_outcome(self, move):
        next_board = deepcopy(self.board)   # self의 board move했을때 다음 board
        next_state = None                   # next_board 의 game_state

        #======================#
        #                      #
        #    게임마다 다르게    #
        #                      #
        #======================#
        # leaf node 인지 (승부/무승부 인지)판별해야함.

        #  4 dir    ↑  ↗   →   ↘
        dir_row = [-1, -1,  0,  1]
        dir_col = [ 0,  1,  1,  1]

        if(self.turn == 0):
            Color = "B"
            next_board[move[1][0]][move[1][1]] = "B"
        else:
            Color = "W"
            next_board[move[1][0]][move[1][1]] = "W"
            
        # who wins
        for dir in range(4):
            five_counter = 0

            row_buff = move[1][0]
            col_buff = move[1][1]
            while(-1 < row_buff and row_buff < len(next_board) and -1 < col_buff and col_buff < len(next_board)):
                # if color match
                if(Color == next_board[row_buff][col_buff]):
                    # counter up
                    five_counter += 1
                    # mov 1 according to dir
                    row_buff += dir_row[dir]
                    col_buff += dir_col[dir]
                else:
                    break

            # opposite direction
            row_buff = move[1][0] - dir_row[dir]
            col_buff = move[1][1] - dir_col[dir]
            while(-1 < row_buff and row_buff < len(next_board) and -1 < col_buff and col_buff < len(next_board)):
                # if color match
                if(Color == next_board[row_buff][col_buff]):
                    # counter up
                    five_counter += 1
                    # mov 1 according to dir
                    row_buff -= dir_row[dir]
                    col_buff -= dir_col[dir]
                else:
                    break

            if (five_counter == 3):
                next_state = self.turn
        
        # draw
        if(next_state == None):
            available = 0
            for row in range(3):
                for col in range(3):
                    if(next_board[row][col] == '.'):
                        available += 1
            if(available == 0):
                next_state = -1

        return [next_board, next_state]



if(__name__ == "__main__"):
    ttt_Board = [['.' for col in range(3)] for row in range(3)]
    node_temp = mcts_node(None, 0, None, ttt_Board, None, 0, 2)

    # tic tac toe 를 tree로 구현한다고 했을때 node의 수는
    # 255,168 와 같습니다.

    while(True):
        print("==========================================")
        # node_temp 가 leaf 가 아닌동안
        while(node_temp.state == None):
            # show board
            #print("Board")
            #for line in node_temp.board:
            #    print(line)
            #print("")
            
            # 매번 다음수를 두기 전 1000회 학습한다.
            best_child_idx = node_temp.mcts_learn(1000, 100000)

            # print 훈수5
            best_child_idx = node_temp.Selection()
            #print("moves : {}".format(node_temp.Playable_options()))
            #print("move{}  : {}".format(best_child_idx, node_temp.child[best_child_idx].move))

            # progress
            # 내가 직접 진행하고 싶으면 다음줄의 주석을 빼시오
            # choice_child = int(input())
            # node_temp = node_temp.child[choice_child]
            node_temp = node_temp.child[best_child_idx]
        
        # print result
        #print("Board")
        #for line in node_temp.board:
        #    print(line)
        #print("")
        print("winner : {}".format(node_temp.state))

        # reset
        node_temp = mcts_node(None, 0, None, ttt_Board, None, 0, 2)
from time import sleep
from math import inf
from random import randint
import copy
import random

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        # self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True

        self.step = 0
        self.isMaxTurn = True
        self.mySearchDepth = 5

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) 
            for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) 
            for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) 
            for row in self.board[6:9]])+'\n')

    # ************************************************************************ #
    #                             Helper Functions                             #
    # ************************************************************************ #

    # Pretty print the board
    def printBoard(self):
        b = copy.deepcopy(self.board)
        empty = [' '] * 11
        for i in range(9):
            b[i].insert(6, ' ')
            b[i].insert(3, ' ')
        b.insert(6, empty)
        b.insert(3, empty)
        s = [[str(e) for e in row] for row in b]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print()

    # Convert index (0-8) to a global offset (0,0) - (2,2).
    # If we want the 4th spot (center) in the 7th local board (lower right),
    # we can do 
    # self.board[self.globalIdx[7][0]+self.localToOffset[4][0]][self.globalIdx[7][1]+self.localToOffset[4][1]]
    # Or just use the getSpot() function below.
    def localToOffset(self, localIndex):
        d1 = 0
        while localIndex > 2:
            d1 += 1
            localIndex -= 3
        return (d1, localIndex)

    # Return a local board as a 1d list from board with a specified global index 
    # (from globalIdx)
    # index: (int, int)   The global index for the local board.
    def getLocalBoard(self, index):
        local = []
        for i in range(9):
            offset = self.localToOffset(i)
            globalCoor = (index[0] + offset[0], index[1] + offset[1])
            local.append(self.board[globalCoor[0]][globalCoor[1]])
        return local

    # Return the specified row in a "localBoard" returned by getLocalBoard()
    def getRow(self, localBoard, rowNum):
        row = localBoard[rowNum*3 : rowNum*3+3].copy()
        return row

    # Return the specified col in a "localBoard" returned by getLocalBoard()
    def getCol(self, localBoard, colNum):
        col = []
        for i in range(3):
            col.append(localBoard[i*3 + colNum])
        return col

    # Get the X row in the local board
    def getX(self, localBoard, legNum):
        list = []
        if legNum == 0:
            list.append(localBoard[0])
            list.append(localBoard[4])
            list.append(localBoard[8])
        elif legNum == 1:
            list.append(localBoard[2])
            list.append(localBoard[4])
            list.append(localBoard[6])
        else:
            raise Exception("Illegal legNum value")
        return list

    # Helper function for checkWinner().
    def checkWinnerHelper(self):
        # For each local board, check to see if there is a winning track occupied
        # by the same player. 
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            # Check to see if there is a row/col that is winning
            for i in range(3):
                row = self.getRow(localBoard, i)
                col = self.getCol(localBoard, i)
                if row[0] != '_' and row[0] == row[1] == row[2]:
                    if row[0] == 'X':
                        return 1
                    else:
                        return -1
                if col[0] != '_' and col[0] == col[1] == col[2]:
                    if col[0] == 'X':
                        return 1
                    else:
                        return -1
            for i in range(2):
                leg = self.getX(localBoard, i)
                if leg[0] != '_' and leg[0] == leg[1] == leg[2]:
                    if leg[0] == 'X':
                        return 1
                    else:
                        return -1
        return 0

    # Determine what row type it is:
    # Return  0 if nothing interesting (nothing that affects utility score)
    # Return  1 if there is a unblocked two-in-a-row for Max
    # Return -1 if there is a unblocked two-in-a-row for Min
    # Return  2 if there is a prevention for Max
    # Return -2 if there is a prevention for Min
    def getRowType(self, row, isMax):
        if (row == ['X', 'X', '_'] or row == ['X', '_', 'X'] 
            or row == ['_', 'X', 'X']) and isMax:
                return 1   # Unblocked two-in-a-row for Max
        elif (row == ['O', 'O', '_'] or row == ['O', '_', 'O'] 
              or row == ['_', 'O', 'O']) and not isMax:
                return -1  # Unblocked two-in-a-row for Min
        elif (row == ['X', 'X', 'O'] or row == ['O', 'X', 'X'] 
              or row == ['X', 'O', 'X']) and not isMax:
                return -2  # Prevention for Min
        elif (row == ['O', 'O', 'X'] or row == ['X', 'O', 'O']
              or row == ['O', 'X', 'O']) and isMax:
                return 2   # Prevention for Max
        return 0

    # @return 
    #   float score: Return the estimated utility score for Max
    def evalPredMax(self):
        score = 0
        
        # print("Evaluating for Max...")
        if self.checkWinner() == 1:
            # If Max wins
            # print("Max could win!!!")
            return 10000

        # print("Max couldn't win; proceed to Rule 2...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            for i in range(3):
                row = self.getRow(localBoard, i)
                col = self.getCol(localBoard, i)
                rowType = self.getRowType(row, True)
                # print("Row {} in local board {}: {}".format(i, local, row))
                if rowType == 1:
                    # print("Type 1 in row {} at local board {}.".format(i, local))
                    score += 500
                elif rowType == 2:
                    # print("Type 2 in row {} at local board {}".format(i, local))
                    score += 100
                colType = self.getRowType(col, True)
                # print("Col {} in local board {}: {}".format(i, local, col))
                if colType == 1:
                    # print("Type 1 in col {} at local board {}.".format(i, local))
                    score += 500
                elif colType == 2:
                    # print("Type 2 in col {} at local board {}.".format(i, local))
                    score += 100
            for i in range(2):
                leg = self.getX(localBoard, i)
                legType = self.getRowType(leg, True)
                # print("Leg {} in local board {}: {}".format(i, local, leg))
                if legType == 1:
                    # print("Type 1 in leg {} at local board {}.".format(i, local))
                    score += 500
                elif legType == 2:
                    # print("Type 2 in leg {} at local board {}.".format(i, local))
                    score += 100
        
        if score != 0:
            # print("Evaluation for Max finished. Score =", score)
            return score
        
        # print("Rule 2 not applicable; proceed to Rule 3...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            if localBoard[0] == 'X':
                # print("Upper left corner taken in local board {}.".format(local))
                score += 30
            if localBoard[2] == 'X':
                # print("Upper right corner taken in local board {}.".format(local))
                score += 30
            if localBoard[6] == 'X':
                # print("Lower left corner taken in local board {}.".format(local))
                score += 30
            if localBoard[8] == 'X':
                # print("Lower right corner taken in local board {}.".format(local))
                score += 30
        
        # if score == 0:
        #     print("Evaluation for Max finished. Nothing interesting is happening now.")
        # else:
        #     print("Evaluation for Max finished. ")
        return score

    # @return 
    #   float score: Return the estimated utility score for Min
    def evalPredMin(self):
        score = 0
        
        # print("Evaluating for Min...")
        if self.checkWinner() == -1:
            # If Min wins
            # print("Min could win!!!")
            return -10000

        # print("Min couldn't win; proceed to Rule 2...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            for i in range(3):
                row = self.getRow(localBoard, i)
                col = self.getCol(localBoard, i)
                rowType = self.getRowType(row, False)
                # print("Row {} in local board {}: {}".format(i, local, row))
                if rowType == -1:
                    # print("Type 1 in row {} at local board {}.".format(i, local))
                    score -= 100
                elif rowType == -2:
                    # print("Type 2 in row {} at local board {}".format(i, local))
                    score -= 500
                colType = self.getRowType(col, False)
                # print("Col {} in local board {}: {}".format(i, local, col))
                if colType == -1:
                    # print("Type 1 in col {} at local board {}.".format(i, local))
                    score -= 100
                elif colType == -2:
                    # print("Type 2 in col {} at local board {}.".format(i, local))
                    score -= 500
            for i in range(2):
                leg = self.getX(localBoard, i)
                legType = self.getRowType(leg, False)
                # print("Leg {} in local board {}: {}".format(i, local, leg))
                if legType == -1:
                    # print("Type 1 in leg {} at local board {}.".format(i, local))
                    score -= 100
                elif legType == -2:
                    # print("Type 2 in leg {} at local board {}.".format(i, local))
                    score -= 500
        
        if score != 0:
            # print("Evaluation for Min finished. Score =", score)
            return score
        
        # print("Rule 2 not applicable; proceed to Rule 3...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            if localBoard[0] == 'O':
                # print("Upper left corner taken in local board {}.".format(local))
                score -= 30
            if localBoard[2] == 'O':
                # print("Upper right corner taken in local board {}.".format(local))
                score -= 30
            if localBoard[6] == 'O':
                # print("Lower left corner taken in local board {}.".format(local))
                score -= 30
            if localBoard[8] == 'O':
                # print("Lower right corner taken in local board {}.".format(local))
                score -= 30
        
        # if score == 0:
        #     print("Evaluation for Min finished. Nothing interesting is happening now.")
        # else:
        #     print("Evaluation for Min finished. ")
        return score

    
    # @return 
    #   float score: Return the estimated utility score for Max
    def evalSuperMax(self):
        raise Exception("stop")
        score = 0
        
        # print("Evaluating for Max...")
        winner = self.checkWinner()
        if winner == 1:
            # If Max wins
            # print("Max could win!!!")
            return 10000
        elif winner == -1:
            return -10000

        # print("Max couldn't win; proceed to Rule 2...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            for i in range(3):
                row = self.getRow(localBoard, i)
                col = self.getCol(localBoard, i)
                rowType = self.getRowType(row, True)
                # print("Row {} in local board {}: {}".format(i, local, row))
                if rowType == 1:
                    # print("Type 1 in row {} at local board {}.".format(i, local))
                    score += 500
                elif rowType == 2:
                    # print("Type 2 in row {} at local board {}".format(i, local))
                    score += 100
                colType = self.getRowType(col, True)
                # print("Col {} in local board {}: {}".format(i, local, col))
                if colType == 1:
                    # print("Type 1 in col {} at local board {}.".format(i, local))
                    score += 500
                elif colType == 2:
                    # print("Type 2 in col {} at local board {}.".format(i, local))
                    score += 100
            for i in range(2):
                leg = self.getX(localBoard, i)
                legType = self.getRowType(leg, True)
                # print("Leg {} in local board {}: {}".format(i, local, leg))
                if legType == 1:
                    # print("Type 1 in leg {} at local board {}.".format(i, local))
                    score += 500
                elif legType == 2:
                    # print("Type 2 in leg {} at local board {}.".format(i, local))
                    score += 100
        
        if score != 0:
            # print("Evaluation for Max finished. Score =", score)
            return score
        
        # print("Rule 2 not applicable; proceed to Rule 3...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            if localBoard[0] == 'X':
                # print("Upper left corner taken in local board {}.".format(local))
                score += 30
            if localBoard[2] == 'X':
                # print("Upper right corner taken in local board {}.".format(local))
                score += 30
            if localBoard[6] == 'X':
                # print("Lower left corner taken in local board {}.".format(local))
                score += 30
            if localBoard[8] == 'X':
                # print("Lower right corner taken in local board {}.".format(local))
                score += 30
        
        # if score == 0:
        #     print("Evaluation for Max finished. Nothing interesting is happening now.")
        # else:
        #     print("Evaluation for Max finished. ")
        return score


    
    # @return 
    #   float score: Return the estimated utility score for Min
    def evalSuperMin(self):
        score = 0
        
        winner = self.checkWinner()
        if winner == -1:
            return -100000
        elif winner == 1:
            return 100000

        # print("Min couldn't win; proceed to Rule 2...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            for i in range(3):
                row = self.getRow(localBoard, i)
                col = self.getCol(localBoard, i)
                rowType = self.getRowType(row, False)
                # print("Row {} in local board {}: {}".format(i, local, row))
                if rowType == -1:
                    # print("Type 1 in row {} at local board {}.".format(i, local))
                    score -= 750
                elif rowType == -2:
                    # print("Type 2 in row {} at local board {}".format(i, local))
                    score -= 500
                elif rowType == 2:
                    score += 100
                elif rowType == 1:
                    score += 500
                colType = self.getRowType(col, False)
                # print("Col {} in local board {}: {}".format(i, local, col))
                if colType == -1:
                    # print("Type 1 in col {} at local board {}.".format(i, local))
                    score -= 750
                elif colType == -2:
                    # print("Type 2 in col {} at local board {}.".format(i, local))
                    score -= 500
                elif colType == 2:
                    score += 100
                elif colType == 1:
                    score += 500
            for i in range(2):
                leg = self.getX(localBoard, i)
                legType = self.getRowType(leg, False)
                # print("Leg {} in local board {}: {}".format(i, local, leg))
                if legType == -1:
                    # print("Type 1 in leg {} at local board {}.".format(i, local))
                    score -= 750
                elif legType == -2:
                    # print("Type 2 in leg {} at local board {}.".format(i, local))
                    score -= 500
                elif legType == 2:
                    score += 100
                elif legType == 1:
                    score += 500
        
        if score != 0:
            # print("Evaluation for Min finished. Score =", score)
            return score
        
        # print("Rule 2 not applicable; proceed to Rule 3...")
        for local in self.globalIdx:
            localBoard = self.getLocalBoard(local)
            if localBoard[0] == 'O':
                # print("Upper left corner taken in local board {}.".format(local))
                score -= 30
            elif localBoard[2] == 'O':
                # print("Upper right corner taken in local board {}.".format(local))
                score -= 30
            elif localBoard[6] == 'O':
                # print("Lower left corner taken in local board {}.".format(local))
                score -= 30
            elif localBoard[8] == 'O':
                # print("Lower right corner taken in local board {}.".format(local))
                score -= 30
        
        # if score == 0:
        #     print("Evaluation for Min finished. Nothing interesting is happening now.")
        # else:
        #     print("Evaluation for Min finished. ")
        return score


    # Return the content in specified spot ('X', 'O' or '_')
    # globalArrIdx: int   Which local board
    # localIdx    : int   Which spot in that local board
    def getSpot(self, globalArrIdx, localIdx):
        if globalArrIdx < 0 or globalArrIdx > 8 or localIdx < 0 or localIdx > 8:
            raise Exception("Illegal board index")
        globalCoord = self.globalIdx[globalArrIdx]
        offset = self.localToOffset(localIdx)
        return self.board[globalCoord[0]+offset[0]][globalCoord[1]+offset[1]]

    # Return a tuple (x,y) translated from globalArrIdx(0-8) and localIdx(0-8)
    def getGlobalIdx(self, globalArrIdx, localIdx):
        globalCoord = self.globalIdx[globalArrIdx]
        offset = self.localToOffset(localIdx)
        return (globalCoord[0]+offset[0], globalCoord[1]+offset[1])

    # Make a move at the specified position
    # globalArrIdx: int     Which local board
    # localIdx    : int     Which spot in that local board
    # move        : string  Stuff to put on the spot
    def play(self, globalArrIdx, localIdx, move):
        if globalArrIdx < 0 or globalArrIdx > 8 or localIdx < 0 or localIdx > 8:
            print("Global: {}, local: {}".format(globalArrIdx, localIdx))
            raise Exception("Illegal board index")
        myBoard = copy.deepcopy(self.board)
        offset = self.localToOffset(localIdx)
        globalCoord = self.globalIdx[globalArrIdx]
        current = myBoard[globalCoord[0]+offset[0]][globalCoord[1]+offset[1]]
        # if current != '_':
        #     raise Exception("Illegal move")
        myBoard[globalCoord[0]+offset[0]][globalCoord[1]+offset[1]] = move
        return myBoard

    # ************************************************************************ #
    #                       UTTT Default Class Functions                       #
    # ************************************************************************ #

    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        if isMax:
            return self.evalPredMax()
        else:
            return self.evalPredMin()


    
    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        if isMax:
            return self.evalSuperMax()
        else:
            return self.evalSuperMin()

    # @return
    #   Return true as long as there are empty spaces on the board. 
    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                         on the board. Return true as long as there are empty 
                         spaces on the board.
        """
        movesLeft = False
        # If there is a '_', that means there is at least 1 empty spot.
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == '_':
                    return True
        return movesLeft

    # @return 
    #   Return 0 if there is no winner.
    #   Return 1 if maxPlayer is the winner.
    #   Return -1 if miniPlayer is the winner.
    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        return self.checkWinnerHelper()
    
    def alphabeta(self, depth, currBoardIdx, alpha, beta, isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        if depth == self.maxDepth:
            value = self.evaluatePredifined(self.isMaxTurn)
            return value
        elif self.checkWinner():
            value = self.evaluatePredifined(self.isMaxTurn)
            return value
        elif not self.checkMovesLeft():
            value = self.evaluatePredifined(self.isMaxTurn)
            return value

        nextDepth = depth + 1
        
        bestValue=0.0
        if isMax:
            bestValue = -inf
        else:
            bestValue = inf
        
        for i in range(9):
            curr = self.getSpot(currBoardIdx, i)
            if curr == '_':  # If this spot is not taken
                self.expandedNodes += 1    # We are considering this one
                self.board = self.play(currBoardIdx, i, 'X' if isMax else 'O')
                # self.printBoard()
                # print()
                if isMax:
                    bestValue = max(bestValue, self.alphabeta(nextDepth, i, alpha, beta, not isMax))
                    if bestValue >= beta:
                        self.board = self.play(currBoardIdx, i, '_')  # Reset
                        return bestValue
                    alpha = max(alpha, bestValue)
                else:
                    bestValue = min(bestValue, self.alphabeta(nextDepth, i, alpha, beta, not isMax))
                    if bestValue <= alpha:
                        self.board = self.play(currBoardIdx, i, '_')  # Reset
                        return bestValue
                    beta = min(beta, bestValue)
                
                self.board = self.play(currBoardIdx, i, '_')  # Reset
        
        return bestValue


    def superAlphabeta(self, depth, currBoardIdx, alpha, beta, isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        if depth == self.mySearchDepth:
            value = self.evaluateDesigned(self.isMaxTurn)
            return value
        elif self.checkWinner():
            value = self.evaluateDesigned(self.isMaxTurn)
            return value
        elif not self.checkMovesLeft():
            value = self.evaluateDesigned(self.isMaxTurn)
            return value

        nextDepth = depth + 1
        
        bestValue=0.0
        if isMax:
            bestValue = -inf
        else:
            bestValue = inf
        
        for i in range(9):
            curr = self.getSpot(currBoardIdx, i)
            if curr == '_':  # If this spot is not taken
                self.expandedNodes += 1    # We are considering this one
                self.board = self.play(currBoardIdx, i, 'X' if isMax else 'O')
                # self.printBoard()
                # print()
                if isMax:
                    bestValue = max(bestValue, self.superAlphabeta(nextDepth, i, alpha, beta, not isMax))
                    if bestValue >= beta:
                        self.board = self.play(currBoardIdx, i, '_')  # Reset
                        return bestValue
                    alpha = max(alpha, bestValue)
                else:
                    bestValue = min(bestValue, self.superAlphabeta(nextDepth, i, alpha, beta, not isMax))
                    if bestValue <= alpha:
                        self.board = self.play(currBoardIdx, i, '_')  # Reset
                        return bestValue
                    beta = min(beta, bestValue)
                
                self.board = self.play(currBoardIdx, i, '_')  # Reset
        
        return bestValue

    
    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float): the bestValue that current player may have
        """

        if depth == self.maxDepth:
            value = self.evaluatePredifined(self.isMaxTurn)
            return value
        elif self.checkWinner():
            value = self.evaluatePredifined(self.isMaxTurn)
            return value
        elif not self.checkMovesLeft():
            value = self.evaluatePredifined(self.isMaxTurn)
            return value

        nextDepth = depth + 1
        
        bestValue=0.0
        if isMax:
            bestValue = -inf
        else:
            bestValue = inf
        
        for i in range(9):
            curr = self.getSpot(currBoardIdx, i)
            if curr == '_':  # If this spot is not taken
                self.expandedNodes += 1    # We are considering this one
                self.board = self.play(currBoardIdx, i, 'X' if isMax else 'O')
                # self.printBoard()
                # print()
                currVal = self.minimax(nextDepth, i, not isMax)
                if isMax:
                    if currVal > bestValue:
                        bestValue = currVal
                else:
                    if currVal < bestValue:
                        bestValue = currVal
                self.board = self.play(currBoardIdx, i, '_')  # Reset
        # print("minimax bestValue:", bestValue)
        return bestValue


    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxDefensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        print("Max First = {}, Max uses minimax = {}, Min uses minimax = {}".format(maxFirst, isMinimaxOffensive, isMinimaxDefensive))
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        expandedNodes = []
        
        winner = 0
        self.isMaxTurn = maxFirst
        currBoardIdx = self.startBoardIdx

        while not winner and self.checkMovesLeft():
            print("isMaxTurn:", self.isMaxTurn)
            currBestMoveIdx = -1
            self.expandedNodes = 0  # Reset
            # currBestValue = inf if not isMaxTurn else -inf
            currBestValue = 0
            for i in range(9):
                curr = self.getSpot(currBoardIdx, i)
                if curr == '_':  # If this spot is not taken
                    self.expandedNodes += 1    # We are considering this one
                    self.board = self.play(currBoardIdx, i, 'X' if self.isMaxTurn else 'O')
                    # self.printBoard()
                    # print()
                    currVal = 0
                    if self.isMaxTurn:
                        if isMinimaxOffensive:
                            currVal = self.minimax(1, i, not self.isMaxTurn)
                            # raise Exception("Stop")
                        else: 
                            currVal = self.alphabeta(1, i, -inf, inf, not self.isMaxTurn)
                    else:  # Our turn! 
                        if isMinimaxDefensive:
                            currVal = self.minimax(1, i, not self.isMaxTurn)
                        else:
                            currVal = self.alphabeta(1, i, -inf, inf, not self.isMaxTurn)
                    # print("currVal:", currVal)
                    if self.isMaxTurn:
                        if currVal > currBestValue:
                            currBestValue = currVal
                            currBestMoveIdx = i
                    else:
                        if currVal < currBestValue:
                            currBestValue = currVal
                            currBestMoveIdx = i
                    self.board = self.play(currBoardIdx, i, '_')  # Reset
            # End of for loop
            
            print("currBestValue:", currBestValue)
            print("currBestMoveIdx:", currBestMoveIdx)
            currBestMove = self.getGlobalIdx(currBoardIdx, currBestMoveIdx)
            bestMove.append(currBestMove)
            bestValue.append(currBestValue)
            self.board = self.play(currBoardIdx, currBestMoveIdx, 'X' if self.isMaxTurn else 'O')
            gameBoards.append(self.board)
            expandedNodes.append(self.expandedNodes)
            currBoardIdx = currBestMoveIdx
            winner = self.checkWinner()
            
            self.step += 1
            print("Step:", self.step)
            self.printBoard()
            print()
            self.isMaxTurn = not self.isMaxTurn
        # End of while loop
        
        return gameBoards, bestMove, expandedNodes, bestValue, winner

    
    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        expandedNodes = []
        
        winner = 0
        # self.isMaxTurn = True  # Change me! 
        self.isMaxTurn = bool(random.getrandbits(1))
        currBoardIdx = self.startBoardIdx

        print("Playing max(offensive, predefined) vs min(defensive, using my own evaluation function)")
        # print("Playing max(offensive, uses my own evaluation function) vs min(defensive, uses my own evaluation function)")
        print("maxFirst = {}, startBoardIdx = {}".format(self.isMaxTurn, self.startBoardIdx))
        print("")
        while not winner and self.checkMovesLeft():
            print("isMaxTurn:", self.isMaxTurn)
            print("currBoardIdx:", currBoardIdx)
            currBestMoveIdx = -1
            self.expandedNodes = 0  # Reset
            
            currBestValue = -inf if self.isMaxTurn else inf
            for i in range(9):
                curr = self.getSpot(currBoardIdx, i)
                if curr == '_':  # If this spot is not taken
                    self.expandedNodes += 1    # We are considering this one
                    self.board = self.play(currBoardIdx, i, 'X' if self.isMaxTurn else 'O')
                    # self.printBoard()
                    # print()
                    currVal = 0
                    if self.isMaxTurn:
                        currVal = self.alphabeta(1, i, -inf, inf, not self.isMaxTurn)
                        # currVal = self.superAlphabeta(1, i, -inf, inf, not self.isMaxTurn)
                    else:
                        currVal = self.superAlphabeta(1, i, -inf, inf, not self.isMaxTurn)
                    # print("currVal:", currVal)
                    if self.isMaxTurn:
                        if currVal > currBestValue:
                            currBestValue = currVal
                            currBestMoveIdx = i
                    else:
                        if currVal < currBestValue:
                            currBestValue = currVal
                            currBestMoveIdx = i
                    self.board = self.play(currBoardIdx, i, '_')  # Reset
            # End of for loop
            
            # print("currBestValue:", currBestValue)
            print("currBestMoveIdx:", currBestMoveIdx)
            currBestMove = self.getGlobalIdx(currBoardIdx, currBestMoveIdx)
            bestMove.append(currBestMove)
            bestValue.append(currBestValue)
            self.board = self.play(currBoardIdx, currBestMoveIdx, 'X' if self.isMaxTurn else 'O')
            gameBoards.append(self.board)
            expandedNodes.append(self.expandedNodes)
            currBoardIdx = currBestMoveIdx
            winner = self.checkWinner()
            
            self.step += 1
            print("Step:", self.step)
            self.printBoard()
            print()
            self.isMaxTurn = not self.isMaxTurn
        # End of while loop

        return gameBoards, bestMove, winner

    
    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        bestMove=[]
        gameBoards=[]
        expandedNodes = []
        
        winner = 0
        # self.isMaxTurn = True  # Change me! 
        self.isMaxTurn = bool(random.getrandbits(1))
        currBoardIdx = self.startBoardIdx

        print("Playing max(human, using their brains) vs min(defensive, using my own evaluation function)")
        print("When it's your turn, enter which spot (0-8) you'd like to play. ")
        print("You will be the X! ")
        
        print("maxFirst = {}, startBoardIdx = {}".format(self.isMaxTurn, self.startBoardIdx))
        self.printBoard()
        print("")
        while not winner and self.checkMovesLeft():
            print("isMaxTurn:", self.isMaxTurn)
            print("currBoardIdx:", currBoardIdx)
            currBestMoveIdx = -1
            self.expandedNodes = 0  # Reset
            
            currBestValue = inf
            if self.isMaxTurn:
                print("Your turn! You're playing on board number {}.".format(currBoardIdx))
                valid = False
                index = -1
                print("0 | 1 | 2")
                print("3 | 4 | 5")
                print("6 | 7 | 8")
                while not valid:
                    index = int(input("Which spot would you want to play on?    "))
                    curr = self.getSpot(currBoardIdx, index)
                    isEmpty = True if curr == '_' else False
                    if index >= 0 and index < 9 and isEmpty:
                        valid = True
                        break
                    print("Invalid index, please enter another one.")
                currBestMove = self.getGlobalIdx(currBoardIdx, index)
                bestMove.append(currBestMove)
                self.board = self.play(currBoardIdx, index, 'X')
                gameBoards.append(self.board)
                currBoardIdx = index

            else:  # For Min
                for i in range(9):
                    curr = self.getSpot(currBoardIdx, i)
                    if curr == '_':  # If this spot is not taken
                        self.expandedNodes += 1    # We are considering this one
                        self.board = self.play(currBoardIdx, i, 'O')
                        # self.printBoard()
                        # print()
                        currVal = 0
                        currVal = self.superAlphabeta(1, i, -inf, inf, not self.isMaxTurn)
                        # print("currVal:", currVal)
                        if currVal < currBestValue:
                                currBestValue = currVal
                                currBestMoveIdx = i
                        self.board = self.play(currBoardIdx, i, '_')  # Reset
                # End of for loop
                
                # print("currBestValue:", currBestValue)
                print("currBestMoveIdx:", currBestMoveIdx)
                currBestMove = self.getGlobalIdx(currBoardIdx, currBestMoveIdx)
                bestMove.append(currBestMove)
                self.board = self.play(currBoardIdx, currBestMoveIdx, 'O')
                gameBoards.append(self.board)
                expandedNodes.append(self.expandedNodes)
                currBoardIdx = currBestMoveIdx
                
            self.step += 1
            print("Step:", self.step)
            self.printBoard()
            print()
            self.isMaxTurn = not self.isMaxTurn
            winner = self.checkWinner()
            if winner == -1:
                print("You're dumb you know.")
            elif winner == 1:
                print("You win. Doesn't matter, you're still ugly.")
        # End of while loop
        totalExpanded = 0
        for number in expandedNodes:
            totalExpanded += number
        print("totalExpanded:", totalExpanded)
        return gameBoards, bestMove, winner



# **************************************************************************** #
#                                     main                                     #
# **************************************************************************** #


if __name__=="__main__":
    uttt=ultimateTicTacToe()

    # maxFirst,isMinimaxOffensive,isMinimaxDefensive
    # gameBoards, bestMove, expandedNodes, bestValue, winner=uttt.playGamePredifinedAgent(True,False,False)
    
    # gameBoards, bestMove, winner = uttt.playGameYourAgent()

    gameBoards, bestMove, winner = uttt.playGameHuman()
    
    # for game in gameBoards:
    #     for row in game:
    #         print(row)
    #     print()
    # print(bestMove)

    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")

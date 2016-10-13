# Ben Lehman


from itertools import chain
from collections import deque
from random import choice
from heapq import heappush, heappop

class Board(object):
    #This class contans the state of the puzzle

    HOLE = " "
    GOAL = [1,2,3,4,5,6,7,8," "]

    def __init__(self, board, hole_location = None):
       self.board = list(chain(board)) if hasattr(board[0], '__iter__') else board
       self.hole = hole_location if hole_location is not None else self.board.index(Board.HOLE)

    # Returns list of possible destinations for the HOLE to go to
    @property
    def possible_moves(self):
        # Up, down
        for dest in (self.hole - 3, self.hole + 3):
            if 0 <= dest < len(self.board):
                yield dest
        # Left, right
        for dest in (self.hole - 1, self.hole + 1):
            if dest // 3 == self.hole // 3:
                yield dest

    #Returns True if current board is equal to the GOAL
    @property
    def solved(self):
        return self.board == self.GOAL

    #Returns new board with hole swapped with the position of the destination
    def new_puzzle(self, destination):
        board = self.board[:]
        board[self.hole], board[destination] = board[destination], board[self.hole]
        return(Board(board, destination))

    # Heuristic that adds to sum based on position of tile compared to the GOAL
    def heuristicOne(self):
        sum = 0
        for i in range(0, len(self.GOAL)):
            tile = self.GOAL[i]
            for j in range(0, len(self.board)):
                if self.board[j] == tile:
                    spread = i % 3 - j % 3
                    # current number is in the correct position
                    if j == i:
                        sum += -2
                    # current number is on the opposite side of its goal position
                    elif spread == 2 or spread == -2:
                        sum += 2
                    # current number is in the same row as its goal position
                    elif j % 3 == i % 3:
                        sum += 1
                    # current number is adjacent to its goal position
                    elif -1 <= (i - j) <= 1:
                        sum += 1
                    else:
                        sum += 2
        return sum

    def heuristicTwo(self):
        sum = 0
        for i in range(0, len(self.GOAL)):
            if i < 8:
                if self.board[i] != i + 1:
                    sum += 1
            else:
                if self.board[i] != " ":
                    sum += 1
        return sum

    def score(self):
        # return 0
        # return self.heuristicOne()
        return self.heuristicTwo()

    def __str__(self):
        return "\n".join(str(self.board[start : start + 3]) for start in range(0, len(self.board), 3))

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        h = 0
        for value, i in enumerate(self.board):
            if i is " ":
                i = 0
            h ^= value << i
        return h

    def __lt__(self, other):
        return self.score() < other.score()

class MakeNode(object):
    def __init__(self, last, prev_holes = None):
        self.last = last
        self.prev_holes = prev_holes or []

    def branch(self, destination):
        return MakeNode(self.last.new_puzzle(destination),
                        self.prev_holes + [self.last.hole])

    def __iter__(self):
        states = [self.last]
        for hole in reversed(self.prev_holes):
            states.append(states[-1].new_puzzle(hole))
        yield from reversed(states)

    def __lt__(self, other):
        return self.last.score() < other.last.score()

class PriorityQueue:
    def __init__(self, start = None):
        self.pqueue = [start] if start is not None else []

    def peek(self):
        return self.pqueue[0]

    def push(self, item):
        heappush(self.pqueue, item)

    def pop(self):
        return heappop(self.pqueue)

    def remove(self, item):
        value = self.pqueue.remove(item)
        heapify(self.pqueue)
        return values is not None

    def __len__(self):
        return len(self.pqueue)


class Solver(object):
    """docstring for Solver"""
    def __init__(self, start):
        self.start = start

    def uninformedSolve(self, limit):
        queue = deque()
        queue.append([MakeNode(self.start)])
        visited = set()
        numRuns = 0
        if self.start.solved:
            return queue.pop()[0]
        while queue:
            current = queue.pop()
            if current[0].last.solved:
                return current[0].last
            for dest in current[0].last.possible_moves:
                cur = current[0].branch(dest)
                if cur.last not in visited:
                    visited.add(cur.last)
                    queue.appendleft([cur])
                limit -= 1
                numRuns += 1
                if limit is 0:
                    print("n")
                    return None
                if cur.last.solved:
                    print(numRuns)
                    return cur

    def informedSolve(self, limit):
        queue = PriorityQueue()
        queue.push([MakeNode(self.start)])
        visited = set()
        numRuns = 0
        if self.start.solved:
            return queue.pop()[0]
        while queue:
            current = queue.pop()
            if current[0].last.solved:
                return current[0].last
            for dest in current[0].last.possible_moves:
                cur = current[0].branch(dest)
                if cur.last not in visited:
                    visited.add(cur.last)
                    queue.push([cur])
                limit -= 1
                numRuns += 1
                if limit is 0:
                    print("n")
                    return None
                if cur.last.solved:
                    print(numRuns)
                    return cur

def random_puzzle():
    p = [0]*9
    poss = [1,2,3,4,5,6,7,8," "]
    for i in range(0, 9):
        k = choice(poss)
        p[i] = k
        poss.remove(k)
    return p

def makeState(nw, n, ne, w, c, e, sw, s, se):
    p = [0]*9
    p[0] = nw
    p[1] = n
    p[2] = ne
    p[3] = w
    p[4] = c
    p[5] = e
    p[6] = sw
    p[7] = s
    p[8] = se
    return p

def testUninformedSearch(init, goal, limit):
    a = Board(init)
    Board.GOAL = goal
    move_seq = Solver(a).uninformedSolve(limit)
    print("Uninformed Test Results:")
    if move_seq is not None:
        for i in list(move_seq):
            print(i)
            print("\n")

def testInformedSearch(init, goal, limit):
    a = Board(init)
    if goal:
        Board.GOAL = goal
    move_seq = Solver(a).informedSolve(limit)
    print("Informed Test Results:")
    if move_seq is not None:
        for i in list(move_seq):
            print(i)
            print("\n")

def sample_maker(size = 100):
    sample = []
    for i in range(size):
        a = random_puzzle()
        sample.append(a)
    return sample

def runTestUninformed(sample, limit = 15000):
    for samp in sample:
        a = Board(samp)
        mov_seq = Solver(a).uninformedSolve(limit)

def runTestInformed(sample, limit = 15000):
    for samp in sample:
        a = Board(samp)
        mov_seq = Solver(a).informedSolve(limit)


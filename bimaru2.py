# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 70:
# 102484 Diogo Ribeiro
# 102516 João Rodrigues
import numpy as np

import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe

def is_boat(value) -> bool:
    return value != '.' and value != None


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, matrix, rows, cols, squares_left_row, squares_left_col):
        self.matrix = matrix
        self.rows = rows    # contar quantos quadrados falta preencher
        self.cols = cols    # com barcos em cada linha / coluna
        self.invalid = False
        self.squares_left_row = squares_left_row    # contar quantos quadrados falta
        self.squares_left_col = squares_left_col    # preencher em cada linha / coluna
    
    def __str__(self) -> str:
        string = ''
        '''#print to debug without None
        m = self.matrix.copy()
        m[m == None] = '_'
        for row in m:
            string += (''.join(map(str, row))) + '\n'
        return string'''
        for row in self.matrix:
            string += (''.join(map(str, row))) + '\n'
        return string
    
    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        if 0 <= row < 10 and 0 <= col < 10:
            return self.matrix[row][col]
        return None
    
    def insert_water(self, row: int, col: int):
        if 0 <= row < 10 and 0 <= col < 10 and self.get_value(row, col) == None:
            self.matrix[row][col] = '.'
            self.squares_left_row[row] -= 1
            self.squares_left_col[col] -= 1
    
    def value(self, row: int, col: int) -> str:
        h = self.adjacent_horizontal_values(row, col)
        v = self.adjacent_vertical_values(row, col)

        if is_boat(h[0]) and is_boat(h[1]):
            return 'm'
        if is_boat(h[0]): # ao lado de um left ou middle
            if col == 9 or h[1] in ('.', 'W') or self.rows[row] == 1:
                return 'r'
        if is_boat(h[1]): # ao lado de um right ou middle
            if col == 0 or h[0] in ('.', 'W') or self.rows[row] == 1:
                return 'l'
        
        if is_boat(v[0]) and is_boat(v[1]):
            return 'm'
        if is_boat(v[0]): # ao lado de um top ou middle
            if row == 9 or v[1] in ('.', 'W') or self.cols[col] == 1:
                return 'b'
        if is_boat(v[1]): # ao lado de um bottom ou middle
            if row == 0 or v[0] in ('.', 'W') or self.cols[col] == 1:
                return 't'
        
        if (h[0] in ('.', 'W') or col == 0) and (h[1] in ('.', 'W') or col == 9) and (v[0] in ('.', 'W') or row == 0) and (v[1] in ('.', 'W') or row == 9):
            return 'c'
        
        return 'x'
    
    def new_value(self, row: int, col: int) -> str:
        h = self.adjacent_horizontal_values(row, col)
        v = self.adjacent_vertical_values(row, col)

        if is_boat(h[0]) and is_boat(h[1]):
            return 'm'
        if is_boat(h[0]): # ao lado de um left ou middle
            if col == 9 or h[1] in ('.', 'W'):
                return 'r'
        if is_boat(h[1]): # ao lado de um right ou middle
            if col == 0 or h[0] in ('.', 'W'):
                return 'l'
        
        if is_boat(v[0]) and is_boat(v[1]):
            return 'm'
        if is_boat(v[0]): # ao lado de um top ou middle
            if row == 9 or v[1] in ('.', 'W'):
                return 'b'
        if is_boat(v[1]): # ao lado de um bottom ou middle
            if row == 0 or v[0] in ('.', 'W'):
                return 't'
        
        if (h[0] in ('.', 'W') or col == 0) and (h[1] in ('.', 'W') or col == 9) and (v[0] in ('.', 'W') or row == 0) and (v[1] in ('.', 'W') or row == 9):
            return 'c'
        
        return 'x'
    
    def insert_boat(self, action):
        (row, col, value) = action
        if 0 <= row < 10 and 0 <= col < 10 and self.get_value(row, col) == None:
            if value == 'x':
                value = self.value(row, col)
            self.matrix[row][col] = value
            self.squares_left_row[row] -= 1
            self.squares_left_col[col] -= 1
            self.rows[row] -= 1
            self.cols[col] -= 1
            self.complete_arround_with_water(row, col)
            if self.full_row(row):
                self.complete_row_with_water(row)
            if self.full_col(col):
                self.complete_col_with_water(col)
        return Board(self.matrix, self.rows, self.cols, self.squares_left_row, self.squares_left_col)

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (self.get_value(row-1, col), self.get_value(row+1, col))

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_value(row, col-1), self.get_value(row, col+1))

    def full_col(self, col: int) -> bool:
        return self.cols[col] == 0
    
    def full_row(self, row: int) -> bool:
        return self.rows[row] == 0
    
    def check_completed(self) -> bool:
        for i in range(0, 10):
            if not self.full_col(i) or not self.full_row(i):
                return False
            #checkar os barcos #TODO
        #self.matrix[self.matrix == None] = '.' #preencher o que esta vazio com agua
        return True
    
    def complete_row_with_water(self, row: int):
        for i in range(0,10):
            if self.get_value(row, i) == None:
                self.insert_water(row, i)
    
    def complete_col_with_water(self, col: int):
        for i in range(0,10):
            if self.get_value(i, col) == None:
                self.insert_water(i, col)
    
    def complete_arround_with_water(self, row: int, col: int):
        value = self.get_value(row, col)

        #valores adjacentes nas diagonais
        self.insert_water(row-1, col-1)
        self.insert_water(row-1, col+1)
        self.insert_water(row+1, col-1)
        self.insert_water(row+1, col+1)

        if value in ('C', 'c'):
            self.insert_water(row-1, col)
            self.insert_water(row, col-1)
            self.insert_water(row, col+1)
            self.insert_water(row+1, col)
        elif value in ('T', 't'):
            self.insert_water(row-1, col)
            self.insert_water(row, col-1)
            self.insert_water(row, col+1)
            self.insert_water(row+2, col-1)
            self.insert_water(row+2, col+1)
        elif value in ('B', 'b'):
            self.insert_water(row, col-1)
            self.insert_water(row, col+1)
            self.insert_water(row+1, col)
            self.insert_water(row-2, col-1)
            self.insert_water(row-2, col+1)
        elif value in ('L', 'l'):
            self.insert_water(row-1, col)
            self.insert_water(row, col-1)
            self.insert_water(row+1, col)
            self.insert_water(row-1, col+2)
            self.insert_water(row+1, col+2)
        elif value in ('R', 'r'):
            self.insert_water(row-1, col)
            self.insert_water(row, col+1)
            self.insert_water(row+1, col)
            self.insert_water(row-1, col-2)
            self.insert_water(row+1, col-2)
    
    def is_boat(self, row: int, col: int):
        return self.get_value(row, col) != '.' and self.get_value(row, col) != 'W' and self.get_value(row, col) != None

    def start_board(self):
        for i in range(0, 10):
            if self.full_row(i):
                self.complete_row_with_water(i)
            if self.full_col(i):
                self.complete_col_with_water(i)
        for i in range(0, 10):
            for j in range(0, 10):
                if self.is_boat(i, j):
                    self.complete_arround_with_water(i, j)
        return self
    
    def complete_boats_row(self, row: int):
        for i in range(0,10):
            if self.get_value(row, i) == None:
                value = self.value(row, i)
                self.matrix[row][i] = value
                self.squares_left_row[row] -= 1
                self.squares_left_col[i] -= 1
                self.rows[row] -= 1
                self.cols[i] -= 1
                self.complete_arround_with_water(row, i)
                if self.full_col(i) and self.squares_left_col[i] != 0:
                    self.complete_col_with_water(i) 
        for i in range(0,10):
            if self.get_value(row, i) == 'x':
                self.matrix[row][i] = self.new_value(row, i)
        return Board(self.matrix, self.rows, self.cols, self.squares_left_row, self.squares_left_col)
    
    def complete_boats_col(self, col: int):
        for i in range(0,10):
            if self.get_value(i, col) == None:
                value = self.value(i, col)
                self.matrix[i][col] = value
                self.squares_left_row[i] -= 1
                self.squares_left_col[col] -= 1
                self.rows[i] -= 1
                self.cols[col] -= 1
                self.complete_arround_with_water(i, col)
                if self.full_row(i) and self.squares_left_row[i] != 0:
                    self.complete_row_with_water(i) 
        for i in range(0,10):
            if self.get_value(i, col) == 'x':
                self.matrix[i][col] = self.new_value(i, col)
        return Board(self.matrix, self.rows, self.cols, self.squares_left_row, self.squares_left_col)

    
    def possible_actions(self):
        for i in range(0, 10):
            if self.rows[i] != 0 and self.squares_left_row[i] != 0 and self.rows[i] == self.squares_left_row[i]:
                return [('complete_boats_row', i)]
            if self.cols[i] != 0 and self.squares_left_col[i] != 0 and self.cols[i] == self.squares_left_col[i]:
                return [('complete_boats_col', i)]
        for row in range(0, 10):
            if not self.full_row(row):
                for col in range(0,10):
                    if self.is_boat(row, col):
                        if self.get_value(row, col) in ('T', 't') and not self.is_boat(row+1, col):
                            return [(row+1, col, 'x')]
                        if self.get_value(row, col) in ('B', 'b') and not self.is_boat(row-1, col):
                            return[(row-1, col, 'x')]
                        if self.get_value(row, col) in ('L', 'l') and not self.is_boat(row, col+1):
                            return[(row, col+1, 'x')]
                        if self.get_value(row, col) in ('R', 'r') and not self.is_boat(row, col-1):
                            return[(row, col-1, 'x')]
                        if self.get_value(row, col) in ('M', 'm'):
                            if '.' in self.adjacent_horizontal_values(row, col) and not self.is_boat(row-1, col) and not self.is_boat(row+1, col):
                                return[(row-1, col, 'x'), (row+1, col, 'x')]
                            if '.' in self.adjacent_vertical_values(row, col) and not self.is_boat(row, col-1) and not self.is_boat(row, col+1):
                                return[(row, col-1, 'x'), (row, col+1, 'x')]
        return []
        '''
        """ações possíveis para um quadrado preenchido por uma ponta"""

        if self.get_value(row,col) in ('T', 't'):
            if row == 8 or self.get_value(row+2,col) == '.':
                self.insert_boat(row+1, col, 'b')
            else
                self.insert_boat(row+1, col, 'm ou b')

        elif self.get_value(row,col) in ('B', 'b'):
            if row == 1 or self.get_value(row-2,col) == '.':
                self.insert_boat(row-1, col, 't')
            else
                self.insert_boat(row+1, col, 'm ou t')

        elif self.get_value(row,col) in ('R', 'r'):
            if col == 1 or self.get_value(row,col-2) == '.':
                self.insert_boat(row, col-1, 'l')
            else
                self.insert_boat(row+1, col, 'm ou l')

        elif self.get_value(row,col) in ('L', 'l'):
            if col == 8 or self.get_value(row,col+2) == '.':
                self.insert_boat(row, col+1, 'r')
            else
                self.insert_boat(row+1, col, 'm ou r')
        '''


        '''
        """ações possíveis para um quadrado preenchido por um meio"""
        
        if self.get_value(row,col) in ('M', 'm'):
            h = self.adjacent_horizontal_values(row, col)
            v = self.adjacent_vertical_values(row, col)
            if '.' in h:
                self.insert_boat(row-1, col, 'm ou t')
                self.insert_boat(row+1, col, 'm ou b')
                
                self.insert_water(row, col-1)
                self.insert_water(row, col+1)

            elif '.' in v:
                self.insert_boat(row, col-1, 'm ou l')
                self.insert_boat(row, col+1, 'm ou r')

                self.insert_water(row-1, col)
                self.insert_water(row+1, col)
        '''


        '''
        if self.squares_left_row[row] == self.rows[row]:
            self.complete_row_with_boats(row) #TODO
        if self.squares_left_col[col] == self.cols[col]:
            self.complete_col_with_boats(col) #TODO
        '''

    def end(self):
        for i in range(0,10):
            for j in range(0,10):
                if self.get_value(i, j) == 'x':
                    self.matrix[i][j] = self.new_value(i, j)
        return self

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        row = sys.stdin.readline().replace('\n', '').split('\t')
        rows = row[1:]
        col = sys.stdin.readline().replace('\n', '').split('\t')
        cols = col[1:]
        rows = [int(x) for x in rows]
        cols = [int(x) for x in cols]

        n = int(sys.stdin.readline().replace('\n', ''))
        matrix = np.full((10, 10), None)
        slrow = [10] * 10
        slcol = [10] * 10
        for i in range (0, n):
            hint = sys.stdin.readline().replace('\n', '').split('\t')
            slrow[int(hint[1])] -= 1
            slcol[int(hint[2])] -= 1
            matrix[int(hint[1])][int(hint[2])] = hint[3]
            if (hint[3] != 'W'):
                rows[int(hint[1])] -= 1
                cols[int(hint[2])] -= 1
        
        return Board(matrix, rows, cols, slrow, slcol).start_board()

    # TODO: outros metodos da classe



class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        state = BimaruState(board)
        super().__init__(state)
        pass

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        return state.board.possible_actions()

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        if len(action) == 0:
            state.board.invalid = True
            return None
        if len(action) == 1 and len(action[0]) == 2:
            if action[0][0] == 'complete_boats_row':
                return BimaruState(state.board.complete_boats_row(action[0][1]))
            if action[0][0] == 'complete_boats_col':
                return BimaruState(state.board.complete_boats_col(action[0][1]))
        if len(action) == 1 and len(action[0]) == 3:
            return BimaruState(state.board.insert_boat(action[0]))
        if len(action) == 2 and action[0][2] in ('t', 'b', 'l', 'r', 'm', 'x') and action[1][2] in ('t', 'b', 'l', 'r', 'm', 'x'):
            state.board.insert_boat(action[0])
            return BimaruState(state.board.insert_boat(action[1]))

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        return state.board.check_completed()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    board = Board.parse_instance()
    bimaru = Bimaru(board)

    b1 = BimaruState(board)
    while True:
        '''print(b1.board)
        print(b1.board.rows)
        print(b1.board.cols)
        print(b1.board.squares_left_row)
        print(b1.board.squares_left_col)
        print(bimaru.actions(b1))'''
        if len(bimaru.actions(b1)) == 0:
            break
        b1 = bimaru.result(b1, bimaru.actions(b1))
   
    print(b1.board.end(), end='')

    '''goal = greedy_search(bimaru)
    print(goal.state.board)'''


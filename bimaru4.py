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


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, matrix, rows, cols, squares_left_row, squares_left_col, barcos):
        self.matrix = matrix
        self.rows = rows    # contar quantos quadrados falta preencher
        self.cols = cols    # com barcos em cada linha / coluna
        self.invalid = False
        self.squares_left_row = squares_left_row    # contar quantos quadrados falta
        self.squares_left_col = squares_left_col    # preencher em cada linha / coluna
        self.barcos = barcos                        # nr de barcos 1,2,3,4 que faltam

    def __str__(self) -> str:
        string = ''
        #print to debug without None           '''retirar para a entrega final'''
        m = self.matrix.copy()
        m[m == None] = '_'
        for row in m:
            string += (''.join(map(str, row))) + '\n'
        return string
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

    def copy_board(self):
        matrix = self.matrix.copy()
        rows = self.rows.copy()
        cols = self.cols.copy()
        squares_left_row = self.squares_left_row.copy()
        squares_left_col = self.squares_left_col.copy()
        barcos = self.barcos.copy()
        return Board(matrix, rows, cols, squares_left_row, squares_left_col, barcos)

    def set_value(self, action):
        (row, col, value) = action
        board = self.copy_board()
        return board.insert_boat(row, col, value)
    
    def insert_boat(self, row: int, col: int, value: str):
        if 0 <= row < 10 and 0 <= col < 10 and self.get_value(row, col) == None:
            self.matrix[row][col] = value
            self.squares_left_row[row] -= 1
            self.squares_left_col[col] -= 1
            self.rows[row][1] += 1
            self.cols[col][1] += 1
            self.complete_arround_with_water(row, col)
            if self.full_row(row):
                self.complete_row_with_water(row)
            if self.full_col(col):
                self.complete_col_with_water(col)

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        return (self.get_value(row-1, col), self.get_value(row+1, col))

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        return (self.get_value(row, col-1), self.get_value(row, col+1))
    
    def is_boat(self, row: int, col: int):
        return self.get_value(row, col) != '.' and self.get_value(row, col) != 'W' and self.get_value(row, col) != None

    def full_col(self, col: int) -> bool:
        return self.cols[col][0] - self.cols[col][1] == 0
    
    def full_row(self, row: int) -> bool:
        return self.rows[row][0] - self.rows[row][1] == 0
    
    def diff_cols(self, col: int):
        return self.cols[col][0] - self.cols[col][1]
    
    def diff_rows(self, row: int):
        return self.rows[row][0] - self.rows[row][1]
    
    def full_boats(self) -> bool:
        return self.barcos[0] == 0 and self.barcos[1] == 0 and self.barcos[2] == 0 and self.barcos[3] == 0
    
    def check_completed(self) -> bool:
        if not self.full_boats:
            return False
        for i in range(0, 10):
            if not self.full_col(i) or not self.full_row(i):
                return False
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

    
    
    def complete_if_possible(self, row: int, col: int): #acho q da pra por mais cenas mas n tou a ver
        value = self.get_value(row, col)
            
        if value in ('T', 't'):
            aux1 = self.get_value(row+1,col)
            aux2 = self.get_value(row+2,col)
            if aux1 == None:
                if row == 8 or aux2 in ('.', 'W') or self.diff_cols(col) == 1 or self.diff_rows(row+1) == 1:
                    self.insert_boat(row+1, col, 'b')
                    self.barcos[1] -= 1
                elif aux2 in ('B', 'b'):
                    self.insert_boat(row+1, col, 'm')
                    self.insert_boat(row+2, col, 'b')
                    self.barcos[2] -= 1
            if (aux1 == None or aux2 == None) and self.get_value(row+3,col) in ('B', 'b'):
                    self.insert_boat(row+1, col, 'm')
                    self.insert_boat(row+2, col, 'm')
                    self.barcos[3] -= 1

        elif value in ('B', 'b'):
            aux1 = self.get_value(row-1,col)
            aux2 = self.get_value(row-2,col)
            if aux1 == None:
                if row == 1 or aux2 in ('.', 'W') or self.diff_cols(col) == 1 or self.diff_rows(row-1) == 1:
                    self.insert_boat(row-1, col, 't')
                    self.barcos[1] -= 1
                elif aux2 in ('T', 't'):
                    self.insert_boat(row-1, col, 'm')
                    self.barcos[2] -= 1
            if (aux1 == None or aux2 == None) and self.get_value(row-3,col) in ('T', 't'):
                    self.insert_boat(row-1, col, 'm')
                    self.insert_boat(row-2, col, 'm')
                    self.barcos[3] -= 1

        elif value in ('L', 'l'):
            aux1 = self.get_value(row,col+1)
            aux2 = self.get_value(row,col+2)
            if aux1 == None:
                if col == 8 or aux2 in ('.', 'W') or self.diff_rows(row) == 1 or self.diff_cols(col+1) == 1:
                    self.insert_boat(row, col+1, 'r')
                    self.barcos[1] -= 1
                elif aux2 in ('R', 'r'):
                    self.insert_boat(row, col+1, 'm')
                    self.barcos[2] -= 1
            if (aux1 == None or aux2 == None) and self.get_value(row,col+3) in ('R', 'r'):
                    self.insert_boat(row, col+1, 'm')
                    self.insert_boat(row, col+2, 'm')
                    self.barcos[3] -= 1

        elif value in ('R', 'r'):
            aux1 = self.get_value(row,col+1)
            aux2 = self.get_value(row,col+2)
            if aux1 == None:
                if col == 1 or aux2 in ('.', 'W') or self.diff_rows(row) == 1 or self.diff_cols(col-1) == 1:
                    self.insert_boat(row, col-1, 'l')
                    self.barcos[1] -= 1
                elif aux2 in ('L', 'l'):
                    self.insert_boat(row, col-1, 'm')
                    self.barcos[2] -= 1
            if (aux1 == None or aux2 == None) and self.get_value(row,col-3) in ('L', 'l'):
                    self.insert_boat(row, col-1, 'm')
                    self.insert_boat(row, col-2, 'm')
                    self.barcos[3] -= 1

        elif value in ('M', 'm'):
            
            if '.' in self.adjacent_horizontal_values(row,col) or 'W' in self.adjacent_horizontal_values(row,col):
                aux_1 = self.get_value(row-1,col)
                aux_2 = self.get_value(row-2,col)
                aux1 = self.get_value(row+1,col)
                aux2 = self.get_value(row+2,col)
                if aux_1 == None and aux1 == None and (self.diff_cols(col) == 2 or (aux_2 in ('.', 'W') and aux2 in ('.', 'W'))):
                    self.insert_boat(row-1, col, 't')
                    self.insert_boat(row+1, col, 'b')
                    self.barcos[2] -= 1
                if aux1 == None and (row == 8 or aux2 in ('.', 'W')
                                     or self.diff_cols(col) == 1 or self.diff_rows(row+1) == 1):
                    self.insert_boat(row+1, col, 'b')
                if aux_1 == None and (row == 1 or aux_2 in ('.', 'W') 
                                      or self.diff_cols(col) == 1 or self.diff_rows(row-1) == 1):
                    self.insert_boat(row-1, col, 't')
            
            elif '.' in self.adjacent_vertical_values(row,col) or 'W' in self.adjacent_vertical_values(row,col):
                aux_1 = self.get_value(row,col-1)
                aux_2 = self.get_value(row,col-2)
                aux1 = self.get_value(row,col+1)
                aux2 = self.get_value(row,col+2)
                if aux_1 == None and aux1 == None and (self.diff_cols(col) == 2 or (aux_2 in ('.', 'W') and aux2 in ('.', 'W'))):
                    self.insert_boat(row-1, col, 'l')
                    self.insert_boat(row+1, col, 'r')
                    self.barcos[2] -= 1
                if aux1 == None and (col == 8 or aux2 in ('.', 'W')
                                     or self.diff_rows(row) == 1 or self.diff_cols(col+1) == 1):
                    self.insert_boat(row, col+1, 'r')
                if aux_1 == None and (col == 1 or aux_2 in ('.', 'W')
                                      or self.diff_rows(row) == 1 or self.diff_cols(col-1) == 1):
                    self.insert_boat(row, col-1, 'l')

    def count_initial_boats(self, row: int, col: int):
        if (self.get_value(row, col) == 'T'):
            if self.is_boat(row+1, col) and self.is_boat(row+2, col) and self.is_boat(row+3, col):
                self.barcos[3] -= 1
            elif self.is_boat(row+1, col) and self.is_boat(row+2, col):
                self.barcos[2] -= 1
            elif self.is_boat(row+1, col):
                self.barcos[1] -= 1
        elif (self.get_value(row, col) == 'L'):
            if self.is_boat(row, col+1) and self.is_boat(row, col+2) and self.is_boat(row, col+3):
                self.barcos[3] -= 1
            elif self.is_boat(row, col+1) and self.is_boat(row, col+2):
                self.barcos[2] -= 1
            elif self.is_boat(row, col+1):
                self.barcos[1] -= 1

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
                    self.count_initial_boats(i , j)
        for i in range(0, 10):
            for j in range(0, 10):
                if self.is_boat(i, j):
                    self.complete_if_possible(i, j)
        return self


    def possible_actions(self):
        boat_size = 4
        if self.barcos[3] == 0:
            if self.barcos[2] == 0:
                boat_size = 2
            else:
                boat_size = 3
        rows_to_test = []
        cols_to_test = []
        for i in range(0,10):
            if self.rows[i][0] >= boat_size:
                rows_to_test += [i]
            if self.cols[i][0] >= boat_size:
                cols_to_test += [i]

        actions = []
        for row in rows_to_test:
            count = 0
            for col in range(0,10):
                if self.is_boat(row, col) or self.get_value(row, col) == None:
                    count += 1
                else:
                    count = 0
                if count >= boat_size:  #col-boat_size+1 é o inicio do barco a contar da esquerda
                    actions += [('insert_row', row, col-boat_size+1, boat_size)]
        for col in cols_to_test:
            count = 0
            for row in range(0,10):
                if self.is_boat(row, col) or self.get_value(row, col) == None:
                    count += 1
                else:
                    count = 0
                if count >= boat_size:  #row-boat_size+1 é o inicio do barco a contar de cima
                    actions += [('insert_col', row-boat_size+1, col, boat_size)]

        return actions


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

        new_rows = []
        new_cols = []
        for i in range(0,10):
            new_rows += [[rows[i], 0]]
            new_cols += [[cols[i], 0]]

        n = int(sys.stdin.readline().replace('\n', ''))
        matrix = np.full((10, 10), None)
        slrow = [10] * 10
        slcol = [10] * 10
        barcos = [4, 3, 2, 1]
        for i in range (0, n):
            hint = sys.stdin.readline().replace('\n', '').split('\t')
            slrow[int(hint[1])] -= 1
            slcol[int(hint[2])] -= 1
            matrix[int(hint[1])][int(hint[2])] = hint[3]
            if (hint[3] != 'W'):
                if (hint[3] == 'C'):
                    barcos[0] -= 1
                new_rows[int(hint[1])][1] += 1
                new_cols[int(hint[2])][1] += 1
        
        return Board(matrix, new_rows, new_cols, slrow, slcol, barcos).start_board()

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



        """ if len(action) == 2:
            if action[0] == 'complete_boats_row':
                return BimaruState(state.board.complete_boats_row(action[1]))
            if action[0] == 'complete_boats_col':
                return BimaruState(state.board.complete_boats_col(action[1]))
        if len(action) == 3:
            return BimaruState(state.board.insert_boat(action))
        if len(action) == 6:
            a = action
            new_board = state.board.insert_boat((a[0], a[1], a[2]))
            return BimaruState(new_board.insert_boat((a[3], a[4], a[5]))) """

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
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    board = Board.parse_instance()
    bimaru = Bimaru(board)

    goal = depth_first_tree_search(bimaru)
    #print(goal.state.board, end='')
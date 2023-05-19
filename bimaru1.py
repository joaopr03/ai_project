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
    
    def insert_boat(self, row: int, col: int, value: str):
        if 0 <= row < 10 and 0 <= col < 10 and self.get_value(row, col) == None:
            self.matrix[row][col] = value
            self.squares_left_row[row] -= 1
            self.squares_left_col[col] -= 1
            self.rows -= 1
            self.cols -= 1
    
    '''def set_value(self, row: int, col: int, value: str):
        """Devolve um novo Board com o novo valor na posição indicada"""
        """Não sei se vai ter grande utilidade"""
        new_matrix = self.matrix.copy()
        new_rows = self.rows.copy()
        new_cols = self.cols.copy()
        new_slrow = self.squares_left_row.copy()
        new_slcol = self.squares_left_col.copy()
        new_matrix[row][col] = value
        new_slrow[row] -= 1
        new_slcol[col] -= 1
        if value != '.':
            new_rows[row] -= 1
            new_cols[col] -= 1
        return Board(new_matrix, new_cols, new_rows, new_slrow, new_slcol)'''

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
        elif value in ('B', 'b'):
            self.insert_water(row, col-1)
            self.insert_water(row, col+1)
            self.insert_water(row+1, col)
        elif value in ('L', 'l'):
            self.insert_water(row-1, col)
            self.insert_water(row, col-1)
            self.insert_water(row+1, col)
        elif value in ('R', 'r'):
            self.insert_water(row-1, col)
            self.insert_water(row, col+1)
            self.insert_water(row+1, col)

    def start_board(self):
        for i in range(0, 10):
            if self.full_row(i):
                self.complete_row_with_water(i)
            if self.full_col(i):
                self.complete_col_with_water(i)
        for i in range(0, 10):
            for j in range(0, 10):
                if self.get_value(i, j) != '.' and self.get_value(i, j) != None:
                    self.complete_arround_with_water(i, j)
        return self
    
    def possible_actions(self):
        '''
        """completar barcos de tamaho 2 nos seguintes casos"""
        if self.get_value(row,col) in ('T', 't'):
            if row == 8 or self.get_value(row+2,col) == '.':
                self.insert_boat(row+1, col, 'b')
        elif self.get_value(row,col) in ('B', 'b'):
            if row == 1 or self.get_value(row-2,col) == '.':
                self.insert_boat(row-1, col, 't')
        elif self.get_value(row,col) in ('R', 'r'):
            if col == 1 or self.get_value(row,col-2) == '.':
                self.insert_boat(row, col-1, 'l')
        elif self.get_value(row,col) in ('L', 'l'):
            if col == 8 or self.get_value(row,col+2) == '.':
                self.insert_boat(row, col+1, 'r')
        '''
        """mais ideias:
            -   comparar squares_left_row e squares_left_col com
                rows e cols se for igual preencher com barcos
            
            -   verificar os valores de rows ou cols que são iguais
                a 4, 3, 2... para ver onde colocar o maior barco"""
 

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
            if (hint[3] == 'W'):
                matrix[int(hint[1])][int(hint[2])] = '.'
            else:
                matrix[int(hint[1])][int(hint[2])] = hint[3]
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
        # TODO
        pass

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

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
    '''board = Board.parse_instance()
    bimaru = Bimaru(board)
    goal = greedy_search(bimaru)
    print(goal.state.board)'''
    # TODO
    pass

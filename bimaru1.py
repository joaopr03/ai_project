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

    def __init__(self, matrix, rows, cols, row_count, col_count):
        self.matrix = matrix
        self.rows = rows
        self.cols = cols
        self.invalid = False
        self.row_count = row_count # contar o nr de quadrados preenchidos
        self.col_count = col_count # sem ser por agua em cada linha / coluna

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.matrix[row][col]
    
    def set_value(self, row: int, col: int, value: str) -> Board:
        """Devolve um novo Board com o novo valor na posição indicada"""
        """Não sei se vai ter grande utilidade"""
        new_matrix = self.matrix.copy()
        new_rows = self.rows.copy()
        new_cols = self.cols.copy()
        new_row_count = self.row_count.copy()
        new_col_count = self.col_count.copy()
        new_matrix[row][col] = value
        if value != '.':
            new_row_count[row] += 1
            new_col_count[col] += 1
        return Board(new_matrix, new_cols, new_rows, new_row_count, new_col_count)

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        if (row == 0):
            return (None, self.matrix[row+1][col])
        if (row == 9):
            return (self.matrix[row-1][col], None)
        return (self.matrix[row-1][col], self.matrix[row+1][col])

    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        if (col == 0):
            return (None, self.matrix[row][col+1])
        if (col == 9):
            return (self.matrix[row][col-1], None)
        return (self.matrix[row][col-1], self.matrix[row][col+1])
    
    def check_completed(self) -> bool:
        for i in range(0, 10):
            if self.cols[i] - self.col_count[i] != 0 or self.rows[i] - self.row_count[i] != 0:
                return False
            #checkar os barcos
        return True
    
    def complete_row_with_water(self, row: int):
        for i in range(0,10):
            if self.matrix[row][i] == None:
                self.matrix[row][i] = '.'
    
    def complete_col_with_water(self, col: int):
        for i in range(0,10):
            if self.matrix[i][col] == None:
                self.matrix[i][col] = '.'
    
    def possible_actions(self):
        # ☠️
        pass


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
        row_count = np.zeros((10,))
        col_count = np.zeros((10,))
        for i in range (0, n):
            hint = sys.stdin.readline().replace('\n', '').split('\t')
            if (hint[3] == 'W'):
                matrix[int(hint[1])][int(hint[2])] = '.'
            else:
                matrix[int(hint[1])][int(hint[2])] = hint[3]
                row_count[int(hint[1])] += 1
                col_count[int(hint[2])] += 1
        
        return Board(matrix, rows, cols, row_count, col_count)

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
    board = Board.parse_instance()
    bimaru = Bimaru(board)
    # TODO
    pass

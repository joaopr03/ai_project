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

    def __init__(self, matrix, rows, cols):
        self.matrix = matrix
        self.rows = rows
        self.cols = cols
        self.rows_completed = np.full(10, False) 
        self.cols_completed = np.full(10, False) 

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.matrix[row][col]

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

    def row_completed(self, row: int) -> bool:
        if not self.rows_completed[row]:
            for i in range(0,10):
                if self.get_value(row, i) == None:
                    return False
            self.rows_completed[row] = True
        return True

    def column_completed(self, col: int) -> bool:
        if not self.cols_completed[col]:    
            for i in range(0,10):
                if self.get_value(i, col) == None:
                    return False
            self.cols_completed[col] = True
        return True

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
        
        for i in range (0, n):
            hint = sys.stdin.readline().replace('\n', '').split('\t')
            matrix[int(hint[1])][int(hint[2])] = hint[3]
        
        return Board(matrix, rows, cols)

    # TODO: outros metodos da classe


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        pass

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""

        table = state.board

        for i in range(0,10):
            if table.column_completed(i):
                pass
            else:
                if table.cols[i] == 0:
                    return "Preencher coluna toda com água"
                count = 0
                for j in range(0,10):
                    if table.get_value(j, i) != "." or table.get_value(j, i) != None:
                        count += 1
                    if count == table.cols[i]:
                        return "Preencher coluna toda com água"

        for i in range(0,10):
            if table.rows_completed(i):
                pass
            else:
                if table.rows[i] == 0:
                    return "Preencher linha toda com água"
                count = 0
                for j in range(0,10):
                    if table.get_value(i, j) != "." or table.get_value(i, j) != None:
                        count += 1
                    if count == table.rows[i]:
                        return "Preencher linha toda com água"

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
        # TODO
        pass

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
    pass

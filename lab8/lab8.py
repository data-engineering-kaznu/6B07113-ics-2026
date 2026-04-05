import numpy as np

# Класс для игры Крестики-нолики
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def print_board(self):
        for i in range(0, 9, 3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6:
                print("-" * 5)

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, move):
        if self.board[move] == ' ':
            self.board[move] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]  # diagonals
        ]
        for condition in win_conditions:
            if self.board[condition[0]] == self.board[condition[1]] == self.board[condition[2]] != ' ':
                return self.board[condition[0]]
        if ' ' not in self.board:
            return 'Tie'
        return None

# Minimax с alpha-beta pruning
def minimax(board, depth, is_maximizing, alpha, beta):
    winner = check_winner(board)
    if winner == 'O':
        return 1
    elif winner == 'X':
        return -1
    elif winner == 'Tie':
        return 0

    if is_maximizing:
        max_eval = -np.inf
        for move in available_moves(board):
            board[move] = 'O'
            eval = minimax(board, depth + 1, False, alpha, beta)
            board[move] = ' '
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in available_moves(board):
            board[move] = 'X'
            eval = minimax(board, depth + 1, True, alpha, beta)
            board[move] = ' '
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def available_moves(board):
    return [i for i, spot in enumerate(board) if spot == ' ']

def check_winner(board):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != ' ':
            return board[condition[0]]
    if ' ' not in board:
        return 'Tie'
    return None

def best_move(board):
    best_score = -np.inf
    move = -1
    for m in available_moves(board):
        board[m] = 'O'
        score = minimax(board, 0, False, -np.inf, np.inf)
        board[m] = ' '
        if score > best_score:
            best_score = score
            move = m
    return move

# Игра
if __name__ == "__main__":
    game = TicTacToe()
    while True:
        game.print_board()
        if game.current_player == 'X':
            move = int(input("Enter your move (0-8): "))
            if not game.make_move(move):
                print("Invalid move")
                continue
        else:
            move = best_move(game.board)
            game.make_move(move)
            print(f"AI plays at {move}")

        winner = game.check_winner()
        if winner:
            game.print_board()
            if winner == 'Tie':
                print("It's a tie!")
            else:
                print(f"{winner} wins!")
            break
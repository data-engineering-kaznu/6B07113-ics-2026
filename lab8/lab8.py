from math import inf


HUMAN = "X"
AI = "O"
EMPTY = " "
WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def print_board(board):
    print()
    for row in range(3):
        start = row * 3
        print(f" {board[start]} | {board[start + 1]} | {board[start + 2]} ")
        if row < 2:
            print("---+---+---")
    print()


def show_positions():
    print("Positions on the board:")
    print()
    for row in range(3):
        start = row * 3
        print(f" {start + 1} | {start + 2} | {start + 3} ")
        if row < 2:
            print("---+---+---")
    print()


def check_winner(board):
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != EMPTY:
            return board[a]
    if EMPTY not in board:
        return "Draw"
    return None


def available_moves(board):
    return [index for index, cell in enumerate(board) if cell == EMPTY]


def minimax(board, depth, maximizing, alpha, beta):
    result = check_winner(board)
    if result == AI:
        return 10 - depth
    if result == HUMAN:
        return depth - 10
    if result == "Draw":
        return 0

    if maximizing:
        best_score = -inf
        for move in available_moves(board):
            board[move] = AI
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = EMPTY
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score

    best_score = inf
    for move in available_moves(board):
        board[move] = HUMAN
        score = minimax(board, depth + 1, True, alpha, beta)
        board[move] = EMPTY
        best_score = min(best_score, score)
        beta = min(beta, best_score)
        if beta <= alpha:
            break
    return best_score


def best_move(board):
    best_score = -inf
    move_choice = None
    for move in available_moves(board):
        board[move] = AI
        score = minimax(board, 0, False, -inf, inf)
        board[move] = EMPTY
        if score > best_score:
            best_score = score
            move_choice = move
    return move_choice


def human_turn(board):
    while True:
        move = input("Enter your move (1-9): ").strip()
        if not move.isdigit():
            print("Please enter a number from 1 to 9.")
            continue

        position = int(move) - 1
        if position not in range(9):
            print("Move must be between 1 and 9.")
            continue
        if board[position] != EMPTY:
            print("That cell is already occupied.")
            continue
        board[position] = HUMAN
        return


def ai_turn(board):
    move = best_move(board)
    if move is not None:
        board[move] = AI
        print(f"AI chooses position {move + 1}.")


def play_game():
    board = [EMPTY] * 9

    print("Tic-tac-toe: human vs AI")
    print(f"You are {HUMAN}, AI is {AI}.")
    show_positions()

    while True:
        print_board(board)
        human_turn(board)
        result = check_winner(board)
        if result:
            break

        ai_turn(board)
        result = check_winner(board)
        if result:
            break

    print_board(board)
    if result == "Draw":
        print("Result: draw.")
    else:
        print(f"Winner: {result}")


if __name__ == "__main__":
    play_game()

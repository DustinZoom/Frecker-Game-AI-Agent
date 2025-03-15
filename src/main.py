# Joey He, Lucas Yu
# COMP30024
# Assignment x





# Board elements representations
EMPTY = 0
LILYPAD = 1
FROG = 2
gridSymbol = {
    EMPTY: "X",    # Empty space
    LILYPAD: "O",  # Lilypad
    FROG: "F"     # Frog
}
# Create an empty 8x8 board
TURN = 1  
BOARD = [[EMPTY for _ in range(8)] for _ in range(8)]





def set_position(board, row, col, element):
    board[row][col] = element
# initialise board as default state
def initialise_board():
    for col in range(8):
        # (row 0 and 1)
        set_position(BOARD, 1, col, LILYPAD)
        if col == 0 or col == 7:
            set_position(BOARD, 0, col, LILYPAD)  # Just lilypads 
        else:
            set_position(BOARD, 0, col, FROG)  # Frogs 
        # (row 7 and 6)
        set_position(BOARD, 6, col, LILYPAD)
        if col == 0 or col == 7:
            set_position(BOARD, 7, col, LILYPAD)  # Just lilypads 
        else:
            set_position(BOARD, 7, col, FROG)  # Frogs 



# ======================Main==============================

# Print the board
for row in BOARD:
    
    initialise_board()

    print(" ".join(gridSymbol[cell] for cell in row))
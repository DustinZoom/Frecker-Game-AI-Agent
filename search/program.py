# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction, BOARD_N
from .utils import render_board
from collections import deque
from copy import deepcopy


def search(
    board: dict[Coord, CellState]
) -> list[MoveAction] | None:
    """
    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `CellState` instances which can be one of
            `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
    
    Returns:
        A list of "move actions" as MoveAction instances, or `None` if no
        solution is possible.
    """
    # Print initial board state
    print("\n=== INITIAL BOARD STATE ===")
    print(render_board(board, ansi=True))
    
    # Find the single red frog
    red_frog = None
    for coord, state in board.items():
        if state == CellState.RED:
            red_frog = coord
            break

    # Initialize BFS
    queue = deque([(red_frog, board, [])])  # (frog_position, board_state, actions_so_far)
    visited = {board_state_hash(red_frog, board)}

    
    print("SEARCHING")
    
    # BFS
    while queue:
        frog_pos, current_board, actions = queue.popleft()
    
        
        # Check if the frog has reached the last row (GOAL STATE)
        if frog_pos.r == BOARD_N - 1:  # Frog is in the last row (row 7)
            print(f"\n=== SOLUTION FOUND! ===")
            print(f"Path length: {len(actions)} moves")
            return actions
        
        # Find all possible moves for the red frog in the current board state
        valid_moves = get_valid_moves(frog_pos, current_board)
        
        for move in valid_moves:
            # Apply the move to get a new board state and frog position
            new_board, new_frog_pos = apply_move(current_board, move)
            state_hash = board_state_hash(new_frog_pos, new_board)
            
            # If we haven't seen this state before, add it to the queue
            if state_hash not in visited:
                visited.add(state_hash)
                queue.append((new_frog_pos, new_board, actions + [move]))
    
    print("NO SOLUTION FOUND")

    # If we've exhausted all possibilities and found no solution
    return None


def board_state_hash(frog_pos: Coord, board: dict[Coord, CellState]) -> frozenset:
    """
    Convert a board state into a hashable representation for the visited set.
    Include the frog position explicitly to ensure unique states.
    """
    # Convert board to a hashable representation
    board_hash = frozenset((str(coord), state) for coord, state in board.items())
    # Return a tuple of frog position and board hash
    return (str(frog_pos), board_hash)


def get_valid_moves(frog_pos: Coord, board: dict[Coord, CellState]) -> list[MoveAction]:
    """
    Find all valid moves for the red frog in the current board state.
    """
    valid_moves = []
    
    # Try all possible single moves in each direction
    for direction in Direction:
        try:
            # Check if direction is allowed (only forward movement)
            direction_is_forward = direction.value.r > 0 or direction.value.r == 0
            
            if not direction_is_forward:
                continue
            
            # First, check for direct movement to adjacent lily pad
            adjacent_pos = frog_pos + direction
            
            # Check for wrapping (replace the is_wrapping_move function)
            if abs(adjacent_pos.r - frog_pos.r) > 1 or abs(adjacent_pos.c - frog_pos.c) > 1:
                continue
            
            # Check if adjacent cell is a lily pad
            if adjacent_pos in board and board[adjacent_pos] == CellState.LILY_PAD:
                valid_moves.append(MoveAction(frog_pos, [direction]))
                continue  # Skip jump-over check if direct move exists
            
            # Now check for jump over moves
            jump_over_pos = adjacent_pos  # Renamed for clarity
            landing_pos = jump_over_pos + direction
            
            # Check for wrapping in the second hop
            if abs(landing_pos.r - jump_over_pos.r) > 1 or abs(landing_pos.c - jump_over_pos.c) > 1:
                continue
            
            # Check if the hop is valid
            if (jump_over_pos in board and 
                (board[jump_over_pos] == CellState.RED or board[jump_over_pos] == CellState.BLUE) and
                landing_pos in board and board[landing_pos] == CellState.LILY_PAD):
                valid_moves.append(MoveAction(frog_pos, [direction]))
                
        except ValueError:
            # This happens if coordinates are out of bounds
            continue
    
    return valid_moves


def apply_move(board: dict[Coord, CellState], move: MoveAction) -> tuple[dict[Coord, CellState], Coord]:
    """
    Apply a move to the board and return the new board state and new frog position.
    """
    new_board = deepcopy(board)
    current_pos = move.coord
    
    # Remove the red frog from its starting position
    del new_board[current_pos]
    
    # Apply the movement direction
    direction = move.directions[0]  # Each move has exactly one direction
    adjacent_pos = current_pos + direction
    
    # Check if it's a direct move to an adjacent lily pad
    if adjacent_pos in board and board[adjacent_pos] == CellState.LILY_PAD:
        current_pos = adjacent_pos
    # Otherwise, it's a jump over move
    else:
        landing_pos = adjacent_pos + direction
        current_pos = landing_pos
    
    # Place the red frog on the final position
    new_board[current_pos] = CellState.RED
    
    return new_board, current_pos
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
    Find a sequence of moves to get the red frog to the bottom row.
    
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

    print("《《SEARCHING》》")
    
    # BFS to find a path to the goal
    solution = find_solution_path(red_frog, board)
    
    if solution:
        print(f"\n=== SOLUTION FOUND! ===")
        print(f"Path length: {len(solution)} moves")
        
        # Print detailed solution analysis
        current_pos = red_frog
        temp_board = deepcopy(board)
        
        print("\n" + "="*60)
        print("DETAILED SOLUTION ANALYSIS")
        print("="*60)
        print(f"Starting position: {red_frog} (row {red_frog.r}, column {red_frog.c})")
        
        # Find final position by applying all moves
        final_board = deepcopy(board)
        final_pos = red_frog
        for move in solution:
            final_board, final_pos = apply_move(final_board, move)
            
        print(f"Final position: {final_pos} (row {final_pos.r}, column {final_pos.c})")
        print(f"Total moves: {len(solution)}")
        print("-"*60)
        
        # Trace through each move in the solution
        print("\nMOVE SEQUENCE:")
        for i, move in enumerate(solution):
            print(f"\nMove {i+1}: {move}")
            print(f"  From: {current_pos} (row {current_pos.r}, column {current_pos.c})")
            
            # Apply the move to get the new position
            temp_board, current_pos = apply_move(temp_board, move)
            
            print(f"  To: {current_pos} (row {current_pos.r}, column {current_pos.c})")
        
        return solution
    
    print("NO SOLUTION FOUND")
    return None


def find_solution_path(start_pos: Coord, start_board: dict[Coord, CellState]) -> list[MoveAction] | None:
    """Find the shortest path from start position to the bottom row using BFS."""
    visited = {board_state_hash(start_pos, start_board)}
    queue = deque([(start_pos, start_board, [])])
    
    while queue:
        pos, board, moves = queue.popleft()
        
        # Goal check: reached the bottom row
        if pos.r == BOARD_N - 1:
            return moves
        
        # Generate and explore valid moves
        for move in get_valid_moves(pos, board):
            corrected_move = MoveAction(pos, move.directions)
            new_board, new_pos = apply_move(board, corrected_move)
            
            state_hash = board_state_hash(new_pos, new_board)
            if state_hash not in visited:
                visited.add(state_hash)
                queue.append((new_pos, new_board, moves + [corrected_move]))
    
    # No path found
    return None


def board_state_hash(frog_pos: Coord, board: dict[Coord, CellState]) -> tuple:
    """Create a hashable representation of the board state."""
    return (str(frog_pos), frozenset((str(coord), state) for coord, state in board.items()))


def get_valid_moves(frog_pos: Coord, board: dict[Coord, CellState]) -> list[MoveAction]:
    """
    Find all valid moves for the red frog in the current board state,
    including multiple jump sequences.
    """
    valid_moves = []
    
    # Check for direct moves to adjacent lily pads
    for direction in Direction:
        # Only allow forward or horizontal movement (not backward)
        if direction in [Direction.Up, Direction.UpLeft, Direction.UpRight]:
            continue
            
        try:
            adjacent_pos = frog_pos + direction
            
            # Check if adjacent cell is a lily pad for direct movement
            if adjacent_pos in board and board[adjacent_pos] == CellState.LILY_PAD:
                valid_moves.append(MoveAction(frog_pos, [direction]))
        except ValueError:
            # Out of bounds
            continue
    
    # Find jump sequences using a helper function
    find_jump_sequences(frog_pos, board, [], valid_moves)
    
    return valid_moves


def find_jump_sequences(current_pos: Coord, board: dict[Coord, CellState], 
                       directions_so_far: list[Direction], result: list[MoveAction]):
    """
    Recursively find all possible jump sequences from the current position.
    """
    # Try each direction for a possible jump
    for direction in Direction:
        # Only allow forward or horizontal movement (not backward)
        if direction in [Direction.Up, Direction.UpLeft, Direction.UpRight]:
            continue
            
        try:
            # Position of the frog/piece we're jumping over
            jump_over_pos = current_pos + direction
            # Position where we'll land after the jump
            landing_pos = jump_over_pos + direction
            
            # Check if this is a valid jump:
            # 1. The jump-over position must contain a frog (red or blue)
            # 2. The landing position must be a lily pad
            is_valid_jump = (
                jump_over_pos in board and
                (board[jump_over_pos] == CellState.RED or board[jump_over_pos] == CellState.BLUE) and
                landing_pos in board and board[landing_pos] == CellState.LILY_PAD
            )
            
            if is_valid_jump:
                # Create a new list with the current direction added
                new_directions = directions_so_far + [direction]
                
                # Add this jump sequence as a valid move
                result.append(MoveAction(current_pos, new_directions))
                
                # Create a temporary board state after this jump
                temp_board = deepcopy(board)
                temp_board[landing_pos] = CellState.RED  # Move frog to landing position
                del temp_board[current_pos]  # Remove frog from starting position
                
                # Recursively look for more jumps from the landing position
                find_jump_sequences(landing_pos, temp_board, new_directions, result)
                
        except ValueError:
            # Out of bounds
            continue


def apply_move(board: dict[Coord, CellState], move: MoveAction) -> tuple[dict[Coord, CellState], Coord]:
    """
    Apply a move to the board and return the new board state and new frog position.
    Handles both direct moves and jump sequences.
    """
    new_board = deepcopy(board)
    current_pos = move.coord
    
    print(f"\n=== Applying move: {move} ===")
    print(f"Starting position: {current_pos} (row {current_pos.r}, column {current_pos.c})")
    
    # Remove the red frog from its starting position
    del new_board[current_pos]
    
    # Apply each direction in the sequence
    for i, direction in enumerate(move.directions):
        print(f"\nStep {i+1}: Applying direction {direction}")
        print(f"  Current position before step: {current_pos} (row {current_pos.r}, column {current_pos.c})")
        
        adjacent_pos = current_pos + direction
        print(f"  Adjacent position: {adjacent_pos} (row {adjacent_pos.r}, column {adjacent_pos.c})")
        
        # For the first step, use the original board to check conditions
        # For subsequent steps, use the updated board state
        reference_board = board if i == 0 else new_board
        
        # Check what's at the adjacent position
        if adjacent_pos in reference_board:
            cell_state = reference_board[adjacent_pos]
            print(f"  Cell at adjacent position contains: {cell_state}")
        else:
            print(f"  Cell at adjacent position is empty or out of bounds")
        
        # Check if it's a direct move to an adjacent lily pad
        if adjacent_pos in reference_board and reference_board[adjacent_pos] == CellState.LILY_PAD:
            print(f"  Direct move to lily pad")
            current_pos = adjacent_pos
        # Otherwise, it's a jump over move
        else:
            try:
                landing_pos = adjacent_pos + direction
                print(f"  Jump attempt. Landing position: {landing_pos} (row {landing_pos.r}, column {landing_pos.c})")
                
                if landing_pos in reference_board:
                    landing_cell = reference_board[landing_pos]
                    print(f"  Cell at landing position contains: {landing_cell}")
                else:
                    print(f"  Cell at landing position is empty or out of bounds")
                
                current_pos = landing_pos
            except ValueError as e:
                print(f"  ERROR: Invalid landing position: {e}")
        
        print(f"  Position after step {i+1}: {current_pos} (row {current_pos.r}, column {current_pos.c})")
    
    # Place the red frog on the final position
    new_board[current_pos] = CellState.RED
    
    print(f"\nFinal position after complete move: {current_pos} (row {current_pos.r}, column {current_pos.c})")
    print("Board after move:")
    print(render_board(new_board, ansi=True))
    
    return new_board, current_pos
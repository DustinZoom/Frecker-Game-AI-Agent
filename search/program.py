# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

# Reference:LLM has been used to improve readability and formatting
# we used some helps from llm to generate a* search algorithm to play around
# however a* is not part of the submission formally

from .core import CellState, Coord, Direction, MoveAction, BOARD_N
from .utils import render_board
from collections import deque
from copy import deepcopy
from heapq import heappush, heappop


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
    
    # BFS search to find a path to the goal
    solution = find_solution_path(red_frog, board)

    # The following line uses A* instead of BFS, just in case grader needs testing it
    # solution = A_find_solution_path(red_frog, board)

    if solution:
        return solution
    
    print("NO SOLUTION FOUND")
    return None

def find_solution_path(start_pos: Coord, start_board: dict[Coord, CellState]) -> list[MoveAction] | None:
    """Find the shortest path using BFS."""

    visited = {board_state_hash(start_pos, start_board)}
    queue = deque([(start_pos, start_board, [])])
    
    while queue:
        pos, board, moves = queue.popleft()

        # Goal check: reached the bottom row
        if pos.r == BOARD_N - 1:
            return moves
        
        # Generate and explore valid moves
        for move in get_valid_moves(pos, board):
            valid_move = MoveAction(pos, move.directions)
            new_board, new_pos = apply_move(board, valid_move)
            
            state_hash = board_state_hash(new_pos, new_board)
            if state_hash not in visited:
                visited.add(state_hash)
                queue.append((new_pos, new_board, moves + [valid_move]))

    
    # No path found
    return None


def board_state_hash(frog_pos: Coord, board: dict[Coord, CellState]) -> tuple:
    """Create a hashable representation of the board state."""
    return (str(frog_pos), frozenset((str(coord), state) for coord, state in board.items()))


def get_valid_moves(frog_pos: Coord, board: dict[Coord, CellState]) -> list[MoveAction]:
    """
    Find all valid moves for the red frog in the current board state,
    including multiple jump sequences.
    Used a DFS approach by recursively calling find_jump_sequences.
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
    
    # Find the jump sequences for this move if we are jumping over frogs.
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
            
            # Check if this is a valid jump-over:
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

    # Remove the red frog from its starting position
    del new_board[current_pos]
    
    # Apply each direction in the sequence
    for i, direction in enumerate(move.directions):
       
        adjacent_pos = current_pos + direction
        
        # For the first step, use the original board to check conditions
        # For subsequent steps, use the updated board state
        reference_board = board if i == 0 else new_board
        
        # Check if it's a direct move to an adjacent lily pad
        if adjacent_pos in reference_board and reference_board[adjacent_pos] == CellState.LILY_PAD:

            current_pos = adjacent_pos

        # Otherwise, it's a jump over move
        else:
            try:
                landing_pos = adjacent_pos + direction
              
                if landing_pos in reference_board and reference_board[landing_pos] == CellState.LILY_PAD:
                  
                    current_pos = landing_pos
                else:
            
                    raise ValueError("Invalid landing: must be a lily pad")
                
                current_pos = landing_pos
            except ValueError as e:
                print(f"  ERROR: Invalid landing position: {e}")
     
    # Place the red frog on the final position
    new_board[current_pos] = CellState.RED
    
    return new_board, current_pos

'''
Below is the attempted A* search algorithm and heuristic function.
Was not adopted in the final search function due to difficulty defining a good heuristic,
as it does not guarantee being admissible in all cases.
'''

def manhattan_distance(pos: Coord) -> int:
    """
    Uses modified a Manhattan distance as a heuristic function to bottom row, considering diagonal moves.
    """
    # Vertical distance to bottom row
    vertical_distance = BOARD_N - 1 - pos.r
    # Prefer central columns for better positioning - less likely to be trapped on edges
    optimal_cols = [BOARD_N // 2, (BOARD_N + 1) // 2]
    horizontal_offset = min(abs(pos.c - mid_col) for mid_col in optimal_cols)
    
    return min(vertical_distance, horizontal_offset)

def A_find_solution_path(start_pos: Coord, start_board: dict[Coord, CellState]) -> list[MoveAction] | None:
    """
    Find the optimal path from start position to the bottom row using A* search
    """
    
    visited = set()
    # Add a unique counter to break ties between states with equal f_scores
    counter = 0
    # Priority queue entries: (f_score, counter, g_score, pos, board, moves)
    g_score = 0
    h_score = manhattan_distance(start_pos)
    f_score = g_score + h_score
    
    # Initialize priority queue with starting state
    queue = [(f_score, counter, g_score, start_pos, start_board, [])]
    
    while queue:
        f_score, _, g_score, pos, board, moves = heappop(queue)
        
        state_hash = board_state_hash(pos, board)
        if state_hash in visited:
            continue
        
        visited.add(state_hash)
        
        # Goal check: reached the bottom row
        if pos.r == BOARD_N - 1:
            return moves
        
        # Generate and explore valid moves
        for move in get_valid_moves(pos, board):
            valid_move = MoveAction(pos, move.directions)
            new_board, new_pos = apply_move(board, valid_move)
            
            if board_state_hash(new_pos, new_board) not in visited:
                counter += 1
                new_g = g_score + 1
                new_h = manhattan_distance(new_pos)
                new_f = new_g + new_h
                
                heappush(queue, (
                    new_f,
                    counter,  # Unique counter to break ties
                    new_g,
                    new_pos,
                    new_board,
                    moves + [valid_move]
                ))
        
    return None
# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
import time

class Agent:
    """
    A Freckers game agent that uses minimax with alpha-beta pruning to make decisions.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """Initialize the agent with its color and a representation of the board."""
        self._color = color
        self._board = {}
        self._init_board()
        
        # Time management
        self._time_remaining = referee.get("time_remaining", 180.0)
        
        # Search parameters
        self._max_depth = 3
        self._move_count = 0
        
         # Position counter - benchmarking only, delete when submit
        self._positions_evaluated = 0
    def _init_board(self):
        """Initialize the board to the starting state."""
        # Initialize all positions to empty
        for r in range(8):
            for c in range(8):
                self._board[Coord(r, c)] = None
        
        # Set up lily pads at the corners
        self._board[Coord(0, 0)] = "LilyPad"
        self._board[Coord(0, 7)] = "LilyPad" 
        self._board[Coord(7, 0)] = "LilyPad"
        self._board[Coord(7, 7)] = "LilyPad"
        
        # Set up lily pads in the second and second-to-last rows
        for r in [1, 6]:
            for c in range(1, 7):
                self._board[Coord(r, c)] = "LilyPad"
        
        # Set up players' frogs
        for c in range(1, 7):
            self._board[Coord(0, c)] = PlayerColor.RED
            self._board[Coord(7, c)] = PlayerColor.BLUE

    def action(self, **referee: dict) -> Action:
        """Choose an action using minimax with alpha-beta pruning."""
        # Update time remaining and move count
        self._time_remaining = referee.get("time_remaining", self._time_remaining)
        self._move_count += 1
        
        # Start timing this move
        start_time = time.time()
        
        # Dynamically adjust search depth based on time remaining
        estimated_moves_left = max(10, 80 - self._move_count)
        time_per_move = self._time_remaining / estimated_moves_left
        
        # Set search depth based on available time
        if time_per_move < 0.5:
            self._max_depth = 1
        elif time_per_move < 1.0:
            self._max_depth = 2
        elif time_per_move < 3.0:
            self._max_depth = 3
        else:
            self._max_depth = 5

        # Get all moves in strategic priority order
        all_moves = self._get_moves(self._board, self._color)
        
        # Set a time budget for this move
        time_budget = min(time_per_move * 0.5, 3.0)
        
        # Apply minimax with alpha-beta pruning
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in all_moves:
            # Stop if we're approaching our time budget
            if time.time() - start_time > time_budget * 0.8:
                break
                
            # Apply the move
            new_board = self._apply_move(self._board, move, self._color)
            
            # Evaluate with minimax
            score = self._minimax(new_board, self._max_depth - 1, alpha, beta, False, start_time, time_budget)
            
            # Update best move if better
            if score > best_score:
                best_score = score
                best_move = move
                
            # Update alpha for pruning
            alpha = max(alpha, best_score)
        
        # Default to first available move if none found
        if best_move is None:
            best_move = all_moves[0]
        
         
        total_time = time.time() - start_time
        print(f"TURN {self._move_count}: {self._color} chose {best_move}")
        print(f"TIME: {total_time:.3f}s / {self._time_remaining:.1f}s remaining, DEPTH: {self._max_depth}")
        print(f"SCORE: {best_score:.1f}, MOVES CONSIDERED: {len(all_moves)}")
        print(f"SCORE: {best_score:.1f}, POSITIONS EVALUATED: {self._positions_evaluated}, NODS/SEC: {self._positions_evaluated/total_time:.1f}")

        return best_move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """Update the internal board state based on the action played."""
        self._time_remaining = referee.get("time_remaining", self._time_remaining)
        
        if isinstance(action, MoveAction):
            source = action.coord
            self._board[source] = None  # Remove from source
            
            # Calculate final position after the move
            final_pos = self._get_final_position(source, action.directions)
            if final_pos:
                self._board[final_pos] = color
        
        elif isinstance(action, GrowAction):
            # Generate lily pads around all frogs of the current player
            self._apply_grow(self._board, color)

    def _minimax(self, board, depth, alpha, beta, is_maximizing, start_time, time_budget):
        """Minimax algorithm with alpha-beta pruning and time check."""
        # Increment the position counter
        self._positions_evaluated += 1
        # Check if we're running out of time
        if time.time() - start_time > time_budget * 0.9:
            return self._evaluate_board(board)
            
        # Terminal conditions
        if self._is_game_over(board):
            return self._evaluate_board(board)
        
        # If depth is exhausted, use quiescence search
        if depth <= 0:
            return self._quiescence_search(board, alpha, beta, is_maximizing, start_time, time_budget, 0)
        
        # Current player
        current_color = self._color if is_maximizing else self._color.opponent
        
        # Get valid moves
        valid_moves = self._get_moves(board, current_color)
        
        # If no valid moves
        if not valid_moves:
            return self._evaluate_board(board)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                # Time check
                if time.time() - start_time > time_budget * 0.85:
                    break
                    
                # Apply move
                new_board = self._apply_move(board, move, current_color)
                
                # Recursive call
                eval = self._minimax(new_board, depth - 1, alpha, beta, False, start_time, time_budget)
                max_eval = max(max_eval, eval)
                
                # Alpha-beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                # Time check
                if time.time() - start_time > time_budget * 0.85:
                    break
                    
                # Apply move
                new_board = self._apply_move(board, move, current_color)
                
                # Recursive call
                eval = self._minimax(new_board, depth - 1, alpha, beta, True, start_time, time_budget)
                min_eval = min(min_eval, eval)
                
                # Alpha-beta pruning
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                
            return min_eval

    def _is_game_over(self, board):
        """Check if the game is over."""
        # Count frogs in destination rows
        red_in_bottom = sum(1 for c in range(8) if board.get(Coord(7, c)) == PlayerColor.RED)
        blue_in_top = sum(1 for c in range(8) if board.get(Coord(0, c)) == PlayerColor.BLUE)
        
        # Game is over if either player has all 6 frogs in destination
        return red_in_bottom == 6 or blue_in_top == 6

    def _evaluate_board(self, board):
        """Evaluate board state from agent's perspective."""
        # Count frogs in destination rows
        red_in_bottom = sum(1 for c in range(8) if board.get(Coord(7, c)) == PlayerColor.RED)
        blue_in_top = sum(1 for c in range(8) if board.get(Coord(0, c)) == PlayerColor.BLUE)
        
        # Base scores
        red_score = red_in_bottom * 100
        blue_score = blue_in_top * 100
           
        # Add progression points
        for r in range(8):
            for c in range(8):
                coord = Coord(r, c)
                if board.get(coord) == PlayerColor.RED:
                    # For RED, higher rows are better
                    red_score += r * 10
                    # Bonus for center columns
                    if 2 <= c <= 5:
                        red_score += 5
                elif board.get(coord) == PlayerColor.BLUE:
                    # For BLUE, lower rows are better
                    blue_score += (7 - r) * 10
                    # Bonus for center columns
                    if 2 <= c <= 5:
                        blue_score += 5
        
        # Return score from agent's perspective
        return red_score - blue_score if self._color == PlayerColor.RED else blue_score - red_score

    def _apply_move(self, board, move, color):
        """Apply move to board and return new board state."""
        new_board = dict(board)
        
        if isinstance(move, MoveAction):
            source = move.coord
            new_board[source] = None  # Remove from source
            
            # Calculate final position after the move
            final_pos = self._get_final_position(source, move.directions, new_board)
            if final_pos:
                new_board[final_pos] = color
            else:
                # If move was invalid, revert
                new_board[source] = color
        
        elif isinstance(move, GrowAction):
            # Apply GROW action
            self._apply_grow(new_board, color)
        
        return new_board

    def _apply_grow(self, board, color):
        """Apply a GROW action to the given board."""
        player_coords = [coord for coord, state in board.items() if state == color]
        for coord in player_coords:
            for direction in Direction:
                try:
                    neighbor = coord + direction
                    if self._is_valid_coord(neighbor) and board.get(neighbor) is None:
                        board[neighbor] = "LilyPad"
                except ValueError:
                    pass

    def _get_final_position(self, start_pos, directions, board=None):
        """Calculate the final position after following a sequence of directions."""
        if board is None:
            board = self._board
            
        curr_pos = start_pos
        
        if len(directions) == 1:
            try:
                dest = curr_pos + directions[0]
                # Check if it's a jump move
                if self._is_valid_coord(dest) and board.get(dest) in [PlayerColor.RED, PlayerColor.BLUE]:
                    dest = dest + directions[0]
                
                # Validate final position
                if self._is_valid_coord(dest):
                    return dest
            except ValueError:
                return None
        else:
            # For multi-jump moves
            for direction in directions:
                try:
                    jump_over = curr_pos + direction
                    landing = jump_over + direction
                    # Validate each step
                    if not (self._is_valid_coord(jump_over) and self._is_valid_coord(landing)):
                        return None
                    curr_pos = landing
                except ValueError:
                    return None
            
            return curr_pos
        
        return None

    def _get_moves(self, board, color):
        """
        Get all valid moves for a player on the given board, in strategic priority order.
        """
        # Find active frogs (not in destination row)
        frogs = [coord for coord, state in board.items() if state == color]
        active_frogs = [coord for coord in frogs if 
                       (color == PlayerColor.RED and coord.r < 7) or 
                       (color == PlayerColor.BLUE and coord.r > 0)]
        
        # Get legal directions and identify forward directions
        legal_directions = self._get_legal_directions(color)
        forward_dirs = ([Direction.Down, Direction.DownLeft, Direction.DownRight] 
                       if color == PlayerColor.RED else 
                       [Direction.Up, Direction.UpLeft, Direction.UpRight])
        
        # Initialize move collections for prioritization
        jump_moves = []
        forward_moves = []
        sideways_moves = []
        grow_move = GrowAction()
        
        # Check for jump moves
        for coord in active_frogs:
            jumps = []
            self._find_jumps_recursive(board, coord, coord, [], jumps, set(), color)
            jump_moves.extend(jumps)
        
        # Check for regular moves
        for coord in active_frogs:
            for direction in legal_directions:
                try:
                    dest = coord + direction
                    if self._is_valid_coord(dest) and board.get(dest) == "LilyPad":
                        move = MoveAction(coord, (direction,))
                        if direction in forward_dirs:
                            forward_moves.append(move)
                        else:
                            sideways_moves.append(move)
                except ValueError:
                    pass
        #  Sort jump moves by length (longer jumps first)
        jump_moves.sort(key=lambda move: len(move.directions), reverse=True)
        
        # Sort forward moves by how far they advance
        forward_moves.sort(key=lambda move: 
            move.coord.r if color == PlayerColor.RED else 7 - move.coord.r, 
            reverse=False)  # Moves from back rank first
        
        # Prioritize GROW earlier in the game, later in midgame
        if self._move_count < 19:  # Early game
            return jump_moves + [grow_move] + forward_moves + sideways_moves
        else:  # Mid/late game
            return jump_moves + forward_moves +  sideways_moves +[grow_move]
     

    def _find_jumps_recursive(self, board, start_pos, current_pos, directions, result, visited, color=None):
        """Find all valid jump sequences recursively."""
        if color is None:
            color = self._color
            
        legal_directions = self._get_legal_directions(color)
        
        for direction in legal_directions:
            try:
                # Calculate positions
                adjacent_pos = current_pos + direction
                landing_pos = adjacent_pos + direction
                
                # Check if in bounds
                if not (self._is_valid_coord(adjacent_pos) and self._is_valid_coord(landing_pos)):
                    continue
                
                # Check if valid jump (over frog onto lily pad)
                if board.get(adjacent_pos) not in [PlayerColor.RED, PlayerColor.BLUE]:
                    continue
                if board.get(landing_pos) != "LilyPad":
                    continue
                
                # Check for cycles
                landing_key = (landing_pos.r, landing_pos.c)
                if landing_key in visited:
                    continue
                
                # Valid jump found
                new_directions = directions + [direction]
                result.append(MoveAction(start_pos, tuple(new_directions)))
                
                # Continue search from landing position
                visited.add(landing_key)
                
                # Create a modified board representation for the recursive call
                temp_board = board.copy()  # Shallow copy is sufficient for the dictionary
                temp_board[landing_pos] = color
                temp_board[current_pos] = None
                
                self._find_jumps_recursive(temp_board, start_pos, landing_pos, new_directions, result, visited, color)
                
            except ValueError:
                continue

    def _get_legal_directions(self, color: PlayerColor) -> list[Direction]:
        """Get legal move directions for a player."""
        if color == PlayerColor.RED:
            return [Direction.Right, Direction.Left, Direction.Down, 
                    Direction.DownLeft, Direction.DownRight]
        else:  # BLUE
            return [Direction.Right, Direction.Left, Direction.Up, 
                    Direction.UpLeft, Direction.UpRight]
                    
    def _is_valid_coord(self, coord):
        """Check if a coordinate is within board bounds."""
        return 0 <= coord.r < 8 and 0 <= coord.c < 8
    


    def _quiescence_search(self, board, alpha, beta, is_maximizing, start_time, time_budget, q_depth):
        """
        Quiescence search to evaluate tactically unstable positions.
        Only considers jumps and destination-reaching moves.
        """
        # Increment position counter
        self._positions_evaluated += 1
        
        # Check time limits
        if time.time() - start_time > time_budget * 0.9:
            return self._evaluate_board(board)
        
        # Get stand-pat score (evaluation without further search)
        stand_pat = self._evaluate_board(board)
        
        # Terminal conditions and depth limit for quiescence
        if self._is_game_over(board) or q_depth >= 5:  # Limit quiescence depth
            return stand_pat
        
        # Update alpha/beta with stand-pat score
        if is_maximizing:
            if stand_pat >= beta:
                return beta  # Beta cutoff
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha  # Alpha cutoff
            beta = min(beta, stand_pat)
        
        # Current player
        current_color = self._color if is_maximizing else self._color.opponent
        
        # Get only tactical moves (jumps and destination-reaching moves)
        tactical_moves = self._get_tactical_moves(board, current_color)
        
        # If no tactical moves, return stand-pat score
        if not tactical_moves:
            return stand_pat
        
        # Search tactical moves
        if is_maximizing:
            max_eval = stand_pat
            for move in tactical_moves:
                # Apply move
                new_board = self._apply_move(board, move, current_color)
                
                # Recursive call
                eval = self._quiescence_search(new_board, alpha, beta, False, start_time, time_budget, q_depth + 1)
                max_eval = max(max_eval, eval)
                
                # Alpha-beta pruning
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
                    
            return max_eval
        else:
            min_eval = stand_pat
            for move in tactical_moves:
                # Apply move
                new_board = self._apply_move(board, move, current_color)
                
                # Recursive call
                eval = self._quiescence_search(new_board, alpha, beta, True, start_time, time_budget, q_depth + 1)
                min_eval = min(min_eval, eval)
                
                # Alpha-beta pruning
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
                    
            return min_eval

    def _get_tactical_moves(self, board, color):
        """Get only tactical moves (jumps and moves to destination row)."""
        # Find active frogs
        frogs = [coord for coord, state in board.items() if state == color]
        active_frogs = [coord for coord in frogs if 
                    (color == PlayerColor.RED and coord.r < 7) or 
                    (color == PlayerColor.BLUE and coord.r > 0)]
        
        # Get legal directions
        legal_directions = self._get_legal_directions(color)
        forward_dirs = ([Direction.Down, Direction.DownLeft, Direction.DownRight] 
                    if color == PlayerColor.RED else 
                    [Direction.Up, Direction.UpLeft, Direction.UpRight])
        
        # Collect tactical moves
        tactical_moves = []
        
        # 1. Get jump moves
        for coord in active_frogs:
            jumps = []
            self._find_jumps_recursive(board, coord, coord, [], jumps, set(), color)
            tactical_moves.extend(jumps)
        
        # 2. Get moves that reach destination row
        destination_row = 7 if color == PlayerColor.RED else 0
        for coord in active_frogs:
            # Only consider frogs one row away from destination
            if (color == PlayerColor.RED and coord.r == 6) or (color == PlayerColor.BLUE and coord.r == 1):
                for direction in forward_dirs:
                    try:
                        dest = coord + direction
                        if (self._is_valid_coord(dest) and 
                            board.get(dest) == "LilyPad" and 
                            dest.r == destination_row):
                            tactical_moves.append(MoveAction(coord, (direction,)))
                    except ValueError:
                        pass
        
        # Sort jump moves by length (longer jumps first)
        tactical_moves.sort(key=lambda move: 
                        len(move.directions) if isinstance(move, MoveAction) else 0, 
                        reverse=True)
        
        return tactical_moves

# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
import random


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """Initialize the agent with its color and a representation of the board."""
        self._color = color
        self._board = {}
        self._init_board()
    
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
        """Choose an action based on jumps > forward > sideways > grow."""
        jump_moves, forward_moves, sideways_moves = self._get_categorized_moves()
        
        if jump_moves:
            return random.choice(jump_moves)
        elif forward_moves:
            return random.choice(forward_moves)
        elif sideways_moves:
            return random.choice(sideways_moves)
        else:
            return GrowAction()

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """Update the internal board state based on the action played."""
        if isinstance(action, MoveAction):
            source = action.coord
            
            # For simple, direct moves
            if len(action.directions) == 1:
                try:
                    dest = source + action.directions[0]
                    # Check if it's a jump move
                    if self._board[dest] in [PlayerColor.RED, PlayerColor.BLUE]:
                        dest = dest + action.directions[0]
                
                    if 0 <= dest.r < 8 and 0 <= dest.c < 8:
                        self._board[dest] = color
                        self._board[source] = None
                except ValueError:
                    pass
            else:
                # For multi-jump moves
                curr_pos = source
                
                # Process each direction in the sequence
                for direction in action.directions:
                    try:
                        jump_over = curr_pos + direction
                        landing = jump_over + direction
                        curr_pos = landing
                    except ValueError:
                        break
                
                self._board[source] = None
                self._board[curr_pos] = color
        
        elif isinstance(action, GrowAction):
            # Generate lily pads around all frogs of the current player
            player_coords = [coord for coord, state in self._board.items() if state == color]
            for coord in player_coords:
                for direction in Direction:
                    try:
                        neighbor = coord + direction
                        if 0 <= neighbor.r < 8 and 0 <= neighbor.c < 8 and self._board.get(neighbor) is None:
                            self._board[neighbor] = "LilyPad"
                    except ValueError:
                        pass

    def _get_legal_directions(self, color: PlayerColor) -> list[Direction]:
        """Get the legal move directions for a player."""
        if color == PlayerColor.RED:
            return [Direction.Right, Direction.Left, Direction.Down, 
                    Direction.DownLeft, Direction.DownRight]
        else:  # BLUE
            return [Direction.Right, Direction.Left, Direction.Up, 
                    Direction.UpLeft, Direction.UpRight]

    def _get_categorized_moves(self):
        """Get valid moves categorized by type: jumps, forward moves, sideways moves."""
        jump_moves = []
        forward_moves = []
        sideways_moves = []
        
        # Find all player's frogs
        player_coords = [coord for coord, state in self._board.items() if state == self._color]
        legal_directions = self._get_legal_directions(self._color)
        
        # Filter out frogs that are already in their destination row
        active_frogs = []
        for coord in player_coords:
            # For RED, destination row is 7; for BLUE, destination row is 0
            if (self._color == PlayerColor.RED and coord.r < 7) or \
            (self._color == PlayerColor.BLUE and coord.r > 0):
                active_frogs.append(coord)

        # Check for regular moves to adjacent lily pads
        for coord in active_frogs:
            for direction in legal_directions:
                try:
                    dest = coord + direction
                    if 0 <= dest.r < 8 and 0 <= dest.c < 8 and self._board.get(dest) == "LilyPad":
                        # Categorize as forward or sideways
                        if self._color == PlayerColor.RED:
                            if direction in [Direction.Down, Direction.DownLeft, Direction.DownRight]:
                                forward_moves.append(MoveAction(coord, (direction,)))
                            else:
                                sideways_moves.append(MoveAction(coord, (direction,)))
                        else:  # BLUE
                            if direction in [Direction.Up, Direction.UpLeft, Direction.UpRight]:
                                forward_moves.append(MoveAction(coord, (direction,)))
                            else:
                                sideways_moves.append(MoveAction(coord, (direction,)))
                except ValueError:
                    pass
        
        # Find valid jump moves
        for coord in active_frogs:
            temp_jumps = self._get_jump_moves(coord)
            jump_moves.extend(temp_jumps)
        
        return jump_moves, forward_moves, sideways_moves

    def _get_jump_moves(self, start_pos: Coord) -> list[MoveAction]:
        """Find all valid jump moves from a position."""
        valid_jumps = []
        visited = set()  # Keep track of positions we've already jumped to
        
        # Search for jumps recursively
        self._find_jumps_recursive(start_pos, start_pos, [], self._board.copy(), valid_jumps, visited)
        
        return valid_jumps

    def _find_jumps_recursive(self, 
                          start_pos: Coord,
                          current_pos: Coord,
                          directions: list,
                          board: dict,
                          result: list,
                          visited: set):
        """Recursively find all valid jump sequences."""
        legal_directions = self._get_legal_directions(self._color)
        
        for direction in legal_directions:
            try:
                # Calculate adjacent position
                adjacent_r = current_pos.r + direction.r
                adjacent_c = current_pos.c + direction.c
                
                # Check if adjacent position is in bounds
                if not (0 <= adjacent_r < 8 and 0 <= adjacent_c < 8):
                    continue
                    
                adjacent_pos = Coord(adjacent_r, adjacent_c)
                
                # Check if there's a frog at the adjacent position
                if board.get(adjacent_pos) not in [PlayerColor.RED, PlayerColor.BLUE]:
                    continue
                
                # Calculate landing position
                landing_r = adjacent_r + direction.r
                landing_c = adjacent_c + direction.c
                
                # Check if landing position is in bounds
                if not (0 <= landing_r < 8 and 0 <= landing_c < 8):
                    continue
                    
                landing_pos = Coord(landing_r, landing_c)
                
                # Check if landing position is a lily pad
                if board.get(landing_pos) != "LilyPad":
                    continue
                
                # Check if we've already visited this position
                landing_key = (landing_r, landing_c)
                if landing_key in visited:
                    continue
                
                # Valid jump found
                new_directions = directions + [direction]
                result.append(MoveAction(start_pos, tuple(new_directions)))
                
                # Track this position as visited
                visited.add(landing_key)
                
                # Create a new board with the frog moved to the landing position
                new_board = board.copy()
                new_board[landing_pos] = self._color
                new_board[current_pos] = None
                
                # Continue search from the landing position
                self._find_jumps_recursive(start_pos, landing_pos, new_directions, new_board, result, visited)
                
            except ValueError:
                continue
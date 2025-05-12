# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
import random
import math
import time


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
        
        # MCTS parameters
        self._max_iterations = 1000
        self._exploration_constant = math.sqrt(2)
    
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
        time_remaining = referee.get("time_remaining", 180.0)
        # Use a fraction of the remaining time, but cap at 2 seconds per move
        time_limit = min(2.0, time_remaining / 20)
        best_action = self._mcts_search(self._board, time_limit=time_limit)
        jump_moves, forward_moves, sideways_moves = self._get_categorized_moves()
        all_moves = jump_moves + forward_moves + sideways_moves + [GrowAction()]
        if best_action is None:
            return random.choice(all_moves)
        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """Update the internal board state based on the action played."""
        # If the action is a move action
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
                
                self._board[source] = None # Removes the frog from the source position
                self._board[curr_pos] = color # Moves the frog to the landing position
        
        # If the action is a grow action
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
        
        # Select frogs that are not in their destination row
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
        
        # Find valid jump-over moves
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

    def _mcts_search(self, root_state, max_iterations=None, time_limit=2.0):
        """Perform MCTS search from the given state."""
        if max_iterations is None:
            max_iterations = self._max_iterations
        jump_moves, forward_moves, sideways_moves = self._get_categorized_moves()
        root = MCTSNode(root_state, color=self._color)
        root.untried_actions = jump_moves + forward_moves + sideways_moves + [GrowAction()]
        start_time = time.time()
        iterations = 0
        while iterations < max_iterations:
            if time.time() - start_time > time_limit:
                break
            node = self._selection(root)
            if node.untried_actions:
                node = self._expansion(node)
            result = self._simulation(node)
            self._backpropagation(node, result)
            iterations += 1
        if root.children:
            return max(root.children, key=lambda x: x.visits).action
        return None

    def _selection(self, node):
        """Select the best child node using UCB1."""
        current = node
        while current.children and not current.untried_actions:
            # Pick the child with the highest UCB1 value
            current = max(current.children, key=lambda x: self._ucb1(x))
        return current

    def _ucb1(self, node):
        """Calculate UCB1 value for a node."""
        if node.visits == 0:
            return float('inf')  # Always explore unvisited nodes first
        return (node.reward / node.visits) + self._exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)

    def _expansion(self, node):
        """Expand the tree by selecting an untried action."""
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        new_board = dict(node.state)
        if isinstance(action, MoveAction):
            source = action.coord
            new_board[source] = None  # Remove from source
            final_pos = self._get_final_position(source, action.directions, new_board)
            if final_pos:
                new_board[final_pos] = node.color
            child_color = node.color.opponent
        elif isinstance(action, GrowAction):
            self._apply_grow(new_board, node.color)
            child_color = node.color.opponent
        else:
            child_color = node.color.opponent
        child = MCTSNode(new_board, parent=node, action=action, color=child_color)
        node.children.append(child)
        return child

    def _simulation(self, node):
        """Simulate a random game from the current state."""
        board = dict(node.state)
        current_color = self._color  # Start with the current player's color
        
        while not self._is_game_over(board):
            # Get all legal actions
            jump_moves, forward_moves, sideways_moves = self._get_categorized_moves_for_board(board, current_color)
            all_moves = jump_moves + forward_moves + sideways_moves + [GrowAction()]
            
            if not all_moves:
                break
                
            # Choose a random action
            action = random.choice(all_moves)
            
            # Apply the action
            if isinstance(action, MoveAction):
                source = action.coord
                board[source] = None
                final_pos = self._get_final_position(source, action.directions, board)
                if final_pos:
                    board[final_pos] = current_color
            elif isinstance(action, GrowAction):
                self._apply_grow(board, current_color)
                
            # Switch player
            current_color = current_color.opponent
        
        # Return the result (1 for win, 0 for loss, 0.5 for draw)
        if self._is_game_over(board):
            if current_color == self._color:
                return 0  # Current player lost
            else:
                return 1  # Current player won
        return 0.5  # Draw

    def _backpropagation(self, node, result):
        """Backpropagate the result up the tree."""
        current = node
        while current:
            current.visits += 1
            current.reward += result
            current = current.parent

    def _get_final_position(self, source, directions, board):
        """Calculate the final position after following a sequence of directions."""
        curr_pos = source
        
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
    
    def _is_valid_coord(self, coord):
        """Check if a coordinate is within board bounds."""
        return 0 <= coord.r < 8 and 0 <= coord.c < 8

    def _is_game_over(self, board):
        """Check if the game is over."""
        # Count frogs in destination rows
        red_in_bottom = sum(1 for c in range(8) if board.get(Coord(7, c)) == PlayerColor.RED)
        blue_in_top = sum(1 for c in range(8) if board.get(Coord(0, c)) == PlayerColor.BLUE)
        
        # Game is over if either player has all 6 frogs in destination
        return red_in_bottom == 6 or blue_in_top == 6

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

    def _get_categorized_moves_for_board(self, board, color):
        """Get valid moves categorized by type for a specific board state."""
        jump_moves = []
        forward_moves = []
        sideways_moves = []
        
        # Find all player's frogs
        player_coords = [coord for coord, state in board.items() if state == color]
        legal_directions = self._get_legal_directions(color)
        
        # Select frogs that are not in their destination row
        active_frogs = []
        for coord in player_coords:
            # For RED, destination row is 7; for BLUE, destination row is 0
            if (color == PlayerColor.RED and coord.r < 7) or \
               (color == PlayerColor.BLUE and coord.r > 0):
                active_frogs.append(coord)

        # Check for regular moves to adjacent lily pads
        for coord in active_frogs:
            for direction in legal_directions:
                try:
                    dest = coord + direction
                    if 0 <= dest.r < 8 and 0 <= dest.c < 8 and board.get(dest) == "LilyPad":
                        # Categorize as forward or sideways
                        if color == PlayerColor.RED:
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
        
        # Find valid jump-over moves
        for coord in active_frogs:
            temp_jumps = self._get_jump_moves_for_board(coord, board, color)
            jump_moves.extend(temp_jumps)
        
        return jump_moves, forward_moves, sideways_moves

    def _get_jump_moves_for_board(self, start_pos: Coord, board: dict, color: PlayerColor) -> list[MoveAction]:
        """Find all valid jump moves from a position for a specific board state."""
        valid_jumps = []
        visited = set()  # Keep track of positions we've already jumped to
        
        # Search for jumps recursively
        self._find_jumps_recursive_for_board(start_pos, start_pos, [], board.copy(), valid_jumps, visited, color)
        
        return valid_jumps

    def _find_jumps_recursive_for_board(self, 
                          start_pos: Coord,
                          current_pos: Coord,
                          directions: list,
                          board: dict,
                          result: list,
                          visited: set,
                          color: PlayerColor):
        """Recursively find all valid jump sequences for a specific board state."""
        legal_directions = self._get_legal_directions(color)
        
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
                new_board[landing_pos] = color
                new_board[current_pos] = None
                
                # Continue search from the landing position
                self._find_jumps_recursive_for_board(start_pos, landing_pos, new_directions, new_board, result, visited, color)
                
            except ValueError:
                continue

class MCTSNode:
    def __init__(self, state, parent=None, action=None, color=None):
        self.state = state  # The board state
        self.parent = parent
        self.action = action  # The action taken to reach this node
        self.children = []
        self.visits = 0
        self.reward = 0
        self.untried_actions = []  # List of actions not yet tried from this node
        self.color = color  # Track which player's turn at this node

# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
import random
import math


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

        # Currently it is selecting random moves
        jump_moves, forward_moves, sideways_moves = self._get_categorized_moves()
        
        # TODO: Implement MCTS
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

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # A Board object
        self.parent = parent
        self.action = action  # The action taken to reach this node
        self.children = []
        self.visits = 0
        self.reward = 0

def ucb1(node, exploration=math.sqrt(2)):
    """a simple version that just picks the best child by UCB1, 
        or returns the current node if it has no children:"""
    if node.visits == 0:
        return float('inf')  # Always explore unvisited nodes first
    return (node.reward / node.visits) + exploration * math.sqrt(math.log(node.parent.visits) / node.visits)

def selection(node):
    """Select the best child node using UCB1."""
    current = node
    while current.children:
        # Pick the child with the highest UCB1 value
        current = max(current.children, key=ucb1)
    return 


def expansion(node):
    """Expand the tree by selecting an untried action."""
    action = node.untried_actions.pop()  # Remove and get an untried action
    new_state = node.state.copy()        # Make a copy of the board
    new_state.apply_action(action)       # Apply the action to the new board
    child = MCTSNode(new_state, parent=node, action=action)
    node.children.append(child)
    return child

def get_legal_actions(board, player_color):
    """Get all legal actions for a player."""
    actions = []
    legal_directions = [Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight] \
        if player_color == PlayerColor.RED else \
        [Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight]

    # Find all player's frogs
    player_coords = [coord for coord, state in board.items() if state == player_color]

    # Filter out frogs that are already in their destination row
    active_frogs = []
    for coord in player_coords:
        if (player_color == PlayerColor.RED and coord.r < 7) or \
           (player_color == PlayerColor.BLUE and coord.r > 0):
            active_frogs.append(coord)

    # Regular moves to adjacent lily pads
    for coord in active_frogs:
        for direction in legal_directions:
            try:
                dest = coord + direction
                if 0 <= dest.r < 8 and 0 <= dest.c < 8 and board.get(dest) == "LilyPad":
                    actions.append(MoveAction(coord, (direction,)))
            except ValueError:
                pass

    # Jump moves (reuse your jump logic if you want, or just add single jumps for now)
    for coord in active_frogs:
        for direction in legal_directions:
            try:
                over = coord + direction
                landing = over + direction
                if (0 <= over.r < 8 and 0 <= over.c < 8 and
                    0 <= landing.r < 8 and 0 <= landing.c < 8 and
                    board.get(over) in [PlayerColor.RED, PlayerColor.BLUE] and
                    board.get(landing) == "LilyPad"):
                    actions.append(MoveAction(coord, (direction, direction)))
            except ValueError:
                pass

    # Always add GrowAction as a possible action
    actions.append(GrowAction())
    return actions

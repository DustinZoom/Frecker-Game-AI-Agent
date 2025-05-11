import time
from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from agent.program import Agent  # Your agent implementation

def analyze_opening_moves(max_depth=8):
    """Run deep searches on the first few moves to identify strong openings."""
    print("Analyzing optimal opening moves...")
    
    # Initialize board to starting position
    board = {}
    for r in range(8):
        for c in range(8):
            board[Coord(r, c)] = None
            
    # Set up lily pads at the corners
    board[Coord(0, 0)] = "LilyPad"
    board[Coord(0, 7)] = "LilyPad" 
    board[Coord(7, 0)] = "LilyPad"
    board[Coord(7, 7)] = "LilyPad"
    
    # Set up lily pads in the second and second-to-last rows
    for r in [1, 6]:
        for c in range(1, 7):
            board[Coord(r, c)] = "LilyPad"
    
    # Set up players' frogs
    for c in range(1, 7):
        board[Coord(0, c)] = PlayerColor.RED
        board[Coord(7, c)] = PlayerColor.BLUE
    
    # Opening sequence to analyze
    opening_sequence = []
    current_board = board.copy()
    
    # Analyze first 6 moves (3 for each player)
    for move_num in range(6):
        print(f"\nAnalyzing move {move_num+1}...")
        
        # Determine current player
        current_color = PlayerColor.RED if move_num % 2 == 0 else PlayerColor.BLUE
        player_name = "RED" if current_color == PlayerColor.RED else "BLUE"
        
        # Create agent and set up
        agent = Agent(current_color, time_remaining=600.0, space_limit=250.0)  # 10 minutes per move
        agent._board = current_board.copy()
        agent._max_depth = max_depth  # Deep search
        
        # Get all possible moves
        possible_moves = agent._get_moves(current_board, current_color)
        
        # Track results for each move
        move_results = []
        
        # Evaluate each possible move
        for i, move in enumerate(possible_moves[:10]):  # Limit to top 10 moves for efficiency
            print(f"  Evaluating {player_name} move option {i+1}: {move}")
            
            # Apply move to get new board state
            next_board = current_board.copy()
            if isinstance(move, MoveAction):
                source = move.coord
                next_board[source] = None
                
                # Calculate final position
                final_pos = agent._get_final_position(source, move.directions, next_board)
                if final_pos:
                    next_board[final_pos] = current_color
            elif isinstance(move, GrowAction):
                # Apply GROW action
                player_coords = [coord for coord, state in next_board.items() if state == current_color]
                for coord in player_coords:
                    for direction in Direction:
                        try:
                            neighbor = coord + direction
                            if 0 <= neighbor.r < 8 and 0 <= neighbor.c < 8 and next_board.get(neighbor) is None:
                                next_board[neighbor] = "LilyPad"
                        except ValueError:
                            pass
            
            # Use agent to evaluate this position (from opponent's perspective)
            agent._board = next_board.copy()
            start_time = time.time()
            evaluation = -agent._evaluate_board(next_board)  # Negate because we're looking from opponent view
            
            # For deeper analysis of promising moves
            if i < 3:  # Only for top 3 candidate moves
                # Run a deeper minimax search
                search_result = agent._minimax(
                    next_board, 
                    max_depth, 
                    float('-inf'), float('inf'), 
                    False,  # opponent's perspective
                    start_time, 
                    600.0  # 10 minute time limit
                )
                # Negate result since it's from opponent's view
                search_evaluation = -search_result
            else:
                search_evaluation = evaluation
                
            elapsed = time.time() - start_time
            
            move_results.append({
                'move': move,
                'eval_score': evaluation,
                'search_score': search_evaluation,
                'time': elapsed
            })
            
            print(f"    Evaluation: {evaluation:.1f}, Search score: {search_evaluation:.1f}, Time: {elapsed:.1f}s")
        
        # Sort by search score
        move_results.sort(key=lambda x: x['search_score'], reverse=True)
        
        # Select best move
        best_move = move_results[0]['move']
        print(f"\n➤ Best {player_name} move: {best_move} (score: {move_results[0]['search_score']:.1f})")
        
        # Add to opening sequence
        opening_sequence.append({
            'player': player_name,
            'move': best_move,
            'score': move_results[0]['search_score']
        })
        
        # Apply best move to current board for next iteration
        if isinstance(best_move, MoveAction):
            source = best_move.coord
            current_board[source] = None
            final_pos = agent._get_final_position(source, best_move.directions, current_board)
            if final_pos:
                current_board[final_pos] = current_color
        elif isinstance(best_move, GrowAction):
            player_coords = [coord for coord, state in current_board.items() if state == current_color]
            for coord in player_coords:
                for direction in Direction:
                    try:
                        neighbor = coord + direction
                        if 0 <= neighbor.r < 8 and 0 <= neighbor.c < 8 and current_board.get(neighbor) is None:
                            current_board[neighbor] = "LilyPad"
                    except ValueError:
                        pass
    
    # Summarize recommended opening sequence
    print("\n=== RECOMMENDED OPENING SEQUENCE ===")
    for i, move_info in enumerate(opening_sequence):
        print(f"{i+1}. {move_info['player']} plays {move_info['move']} (score: {move_info['score']:.1f})")
    
    return opening_sequence

if __name__ == "__main__":
    opening_sequence = analyze_opening_moves()
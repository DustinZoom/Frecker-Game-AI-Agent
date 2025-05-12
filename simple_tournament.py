# simple_tournament.py
import subprocess
import sys
import re
import time
from collections import defaultdict

def run_tournament(agent1_module, agent2_module, num_games=20):
    """Run a tournament using the referee module with minimal output"""
    print(f"Starting tournament: {agent1_module} vs {agent2_module}")
    print(f"Number of games: {num_games}")
    
    # Statistics tracking
    wins = {"RED": 0, "BLUE": 0}
    agent_wins = {agent1_module: 0, agent2_module: 0}
    draws = 0
    move_counts = []
    game_times = []
    
    for game_num in range(num_games):
        # Decide which agent plays as which color
        if game_num % 2 == 0:
            red_module = agent1_module
            blue_module = agent2_module
        else:
            red_module = agent2_module
            blue_module = agent1_module
            
        # Current matchup
        current_agents = {
            "RED": red_module,
            "BLUE": blue_module
        }
        
        print(f"\nGame {game_num+1}/{num_games}: {red_module} (RED) vs {blue_module} (BLUE)")
        
        # Run the referee
        start_time = time.time()
        
        # Build the referee command (minimal verbosity)
        cmd = [
            sys.executable, 
            "-m", 
            "referee", 
            red_module, 
            blue_module,
            "--verbosity", "1"  # Minimal verbosity
        ]
        
        try:
            # Run the referee process
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5-minute timeout
            )
            
            # Game duration
            duration = time.time() - start_time
            game_times.append(duration)
            
            # Extract the result
            output = result.stdout + "\n" + result.stderr  # Check both stdout and stderr
            
            # Parse result and move count
            winner = None
            move_count = 0
            
            # Look for result line
            for line in output.splitlines():
                if "result:" in line:
                    result_text = line.split("result:")[1].strip()
                    if "draw" in result_text.lower():
                        winner = "DRAW"
                    elif "player 1" in result_text.lower() or "red" in result_text.lower():
                        winner = "RED"
                    elif "player 2" in result_text.lower() or "blue" in result_text.lower():
                        winner = "BLUE"
                    else:
                        # Check for module names in result
                        if red_module in result_text.lower():
                            winner = "RED"
                        elif blue_module in result_text.lower():
                            winner = "BLUE"
                    
                # Try to extract move count from commentary
                match = re.search(r"turn (\d+)", line)
                if match:
                    move_number = int(match.group(1))
                    move_count = max(move_count, move_number)
            
            # If we still couldn't determine the winner, check for error messages
            if winner is None:
                if "illegal action" in output.lower():
                    # If one player made an illegal move, the other wins
                    if "red" in output.lower() and "illegal action" in output.lower():
                        winner = "BLUE"
                    elif "blue" in output.lower() and "illegal action" in output.lower():
                        winner = "RED"
                    
            # Still no winner? Check exit code
            if winner is None:
                if result.returncode == 0:
                    # Referee exited successfully, but we couldn't parse the winner
                    # Check the last few lines for clues
                    last_few_lines = "\n".join(output.splitlines()[-10:])
                    if "red" in last_few_lines.lower() and not "blue" in last_few_lines.lower():
                        winner = "RED"
                    elif "blue" in last_few_lines.lower() and not "red" in last_few_lines.lower():
                        winner = "BLUE"
                    else:
                        # Default to draw if we can't determine winner
                        winner = "DRAW"
                        print("WARNING: Could not determine winner, assuming draw")
                        print(f"Last output: {last_few_lines}")
                
            # Update statistics
            if move_count > 0:
                move_counts.append(move_count)
                
            if winner == "DRAW":
                draws += 1
                result_str = "DRAW"
            elif winner in ["RED", "BLUE"]:
                wins[winner] += 1
                winning_agent = current_agents[winner]
                agent_wins[winning_agent] += 1
                result_str = f"{winner} ({winning_agent}) WINS"
            else:
                result_str = "UNKNOWN (parse error)"
                print("WARNING: Could not parse game result")
                # Print last few lines of output for debugging
                print("Last output lines:")
                for line in output.splitlines()[-5:]:
                    print(f"  > {line}")
            
            print(f"Result: {result_str}")
            print(f"Game length: {move_count} moves, {duration:.1f} seconds")
            
        except subprocess.TimeoutExpired:
            print("Game timed out after 5 minutes")
            # Count as a draw
            draws += 1
            
        except Exception as e:
            print(f"Error running game: {e}")
    
    # Calculate statistics
    total_games = wins["RED"] + wins["BLUE"] + draws
    if total_games > 0:
        red_win_rate = wins["RED"] / total_games * 100
        blue_win_rate = wins["BLUE"] / total_games * 100
        draw_rate = draws / total_games * 100
    else:
        red_win_rate = blue_win_rate = draw_rate = 0
    
    avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
    avg_time = sum(game_times) / len(game_times) if game_times else 0
    
    # Print tournament summary
    print("\n" + "="*50)
    print("TOURNAMENT RESULTS")
    print("="*50)
    print(f"Games played: {total_games}")
    print(f"RED wins: {wins['RED']} ({red_win_rate:.1f}%)")
    print(f"BLUE wins: {wins['BLUE']} ({blue_win_rate:.1f}%)")
    print(f"Draws: {draws} ({draw_rate:.1f}%)")
    print("-"*50)
    print("Agent performance:")
    for agent, win_count in agent_wins.items():
        print(f"  {agent}: {win_count} wins ({win_count/total_games*100:.1f}%)")
    print("-"*50)
    print(f"Average game length: {avg_moves:.1f} moves")
    print(f"Average game duration: {avg_time:.1f} seconds")
    print("="*50)
    
    return {
        "wins": wins,
        "agent_wins": agent_wins,
        "draws": draws,
        "total_games": total_games,
        "avg_moves": avg_moves,
        "avg_time": avg_time
    }

if __name__ == "__main__":
    # Default agent modules
    agent1 = "agent"
    agent2 = "random_agent"
    num_games = 20
    
    # Parse command line arguments
    if len(sys.argv) > 2:
        agent1 = sys.argv[1]
        agent2 = sys.argv[2]
    if len(sys.argv) > 3:
        num_games = int(sys.argv[3])
    
    # Run the tournament
    run_tournament(agent1, agent2, num_games)
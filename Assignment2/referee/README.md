The referee
=========================

It is the orchestrator of a match: 
- Kicks things off
- Enforces the rules
- Keeps the clock and turn-loop ticking
- Reports the result
We don't really need to understand everything of it.

Referee
------------------------
1. Entry point & configuration
    - `__main__.py` helps us run the package as a program.
        - Hands over control to the referee logic if we do `python -m referee`
        - For detailed guidance on referee usage/command line options, run: `python -m referee --help`
    - `options.py` parses command-line flags for the core referee code.
2. Core referee logic
    - `run.py` contains the high-level game loop.
        1. Initialise the game state
        2. Launch each agent in a sandboxed subprocess
        3. Alternate asking them for moves
        4. Apply the move to the game
        5. Check for win/loss/draw or timeouts
        6. Log the outcome and shut everything down
3. Logging & housekeeping
    - `log.py` centralises all the printouts, timestamps, and error reports, so to get consistent formatting.
4. `__init__.py` files
    - Mark directories as Python "packages" to import methods like `referee.run`


The agent package 
------------------------
It is responsible for packaging up the "agent" code and run it in a safe subprocess.
1. `subprocess.py`
    - Launches the agent's main program in its own child process, tracks its stdout/stderr, and enforces time-limits or memory limits. 
2. `io.py`
    - Provides low-level routines for parsing the turn-by-turn messages.
3. `client.py`
    - Builds on `io.py` for the agent code to receive the board and return a move.
4. `resources.py` 
    - Houses static data and helper functions for the agent.

File Structure
--------------
- ./               Core referee logic
- agent/           Player agent
- game/            
- server/
     

Author
------
Yongyou (Lucas) Yu, Zhaoyu (Joey) He

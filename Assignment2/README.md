COMP30024 Project Part B
=========================

In this project, we aim to develop an agent to play the full two-player version of Freckers.

Referee terminal command line:

python -m referee agent agentminiv1

python -m referee agent agent_naive

python -m referee agent agenttest

python -m referee agenttest agent 

python -m referee agenttest agent_naive 

python -m referee agenttest agentminiv1

python -m referee agenttest agentminiv1






more information during gameplay
    python -m referee -v 3 agent agent
set a delay between moves to better visualize what's happening:
    python -m referee -w 0.5 agent agent
    python -m referee -w 1.5 agent agent
combine these flags:
    python -m referee -v 3 -w 0.5 agent agent
    
Usage
-----
1. Make sure the referee module and the agent module are under the same (current) working directory.
2. Then `cd` into Assignment2.
3. Run the referee (**don't edit the referee file!**): 
    - `python -m referee agent agent`
    - `python -m referee <red module> <blue module>`
4. To read about the referee to assist with visualising, testing the work, and resource usage.
    - `python -m referee --help`

File Structure
--------------
- agent/           Player code
- referee/         Game engine
- tests/           TBD

Program constraints
-------------------
The max computation time limit is **180 seconds per player, per game**, measured in accumulated CPU time.
The max memory usage is **250MB per player, per game**.



Author
------
Yongyou (Lucas) Yu, Zhaoyu (Joey) He

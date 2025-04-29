COMP30024 Project Part B
=========================

In this project, we aim to develop an agent to play the full two-player version of Freckers.

Usage
-----
1. Make sure the referee module and the agent module are under the same (current) working directory.
2. Then cd into Assignment2.
3. Run the referee ($\underline{don't edit the referee file!}$): 
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
The max computation time limit is $\textbf{180 seconds per player, per game}$, measured in accumulated CPU time.
The max memory usage is $\textbf{250MB per player, per game}$.



Author
------
Yongyou (Lucas) Yu, Zhaoyu (Joey) He

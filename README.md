# Freckers AI Agent

> A research-driven AI agent using Minimax search with custom optimizations

**University of Melbourne | COMP30024 Artificial Intelligence**
**Team: Yongyou (Lucas) Yu, Zhaoyu (Joey) He**
**Final Grade: 29/30**

---

## Project Overview

This README provides a high-level overview of the project and key achievements. For complete technical details, algorithm analysis, and experimental results, see:
- **[report.pdf](report.pdf)** - Full technical report with complexity analysis and benchmarks
- **[agent/program.py](agent/program.py)** - Core implementation
- **[AI_2025_Project_PartB.pdf](AI_2025_Project_PartB.pdf)** - Original assignment specification with Frecker game rules

---

## Feedbacks from Teaching Team

 29/30 in total
 
  **[Grades](grades.png)**

 'A top-tier submission that goes well beyond the baseline requirements.'

 **[Feedback](feedback.png)**

 
## Key Highlights

**Research Assignment:** This was an open-ended research project where students were given the game rules and constraints but no guidance on which algorithms to use. 

**Skills Demonstrated:**
- Adversarial search algorithm design and optimization
- Independent research 
- Performance benchmarking and systematic testing
- Iterative development with empirical validation

**Technical Achievement:**
- 100% win rate against baseline minimax opponents
- Search depth 5-7 (vs. baseline depth 3) under same time constraints
- 15-20% reduction in states explored through optimization
- Outperformed Monte Carlo Tree Search implementation
---

## Technical Implementation

### Research Process

The assignment provided no algorithmic guidance - we had to:
1. Research game-playing algorithms independently
2. Select appropriate techniques for the domain
3. Design and implement custom optimizations
4. Validate improvements through systematic testing

After researching approaches including Minimax, Monte Carlo Tree Search, and techniques from championship programs like Chinook (checkers) and Stockfish (chess), we selected Minimax with alpha-beta pruning as the foundation and developed three custom enhancements.

### Core Algorithm: Minimax with Alpha-Beta Pruning

Standard adversarial search that recursively evaluates future game states, selecting moves that maximize position value while assuming optimal opponent play.

**Evaluation Function:**
- Frog reaches destination: +100 points
- Each row advanced: +10 points
- Center column position: +5 points

This simple linear evaluation outperformed complex alternatives we tested (see Experiments section).

### Enhancement 1: Optimized Move Ordering

**Motivation:** Alpha-beta pruning efficiency depends critically on move ordering. Exploring strong moves first enables more aggressive pruning.

**Implementation:**
1. Jump sequences (sorted by length - longer first)
2. Forward moves (sorted by proximity to goal)
3. Sideways moves
4. Grow action (prioritized early game, normal priority after turn 19)

**Impact:** significant reduction in states explored, enabling deeper search in same time budget.

### Enhancement 2: Quiescence Search

**Motivation:** Stopping search at arbitrary depth can miss critical tactical sequences. Inspired by Chinook's approach.

**Implementation:** When base depth reached, continue searching up to 5 additional levels if "tactical moves" available:
- Jump sequences
- Frogs in second-to-last row (about to score)

**Impact:** Effective search depth increased from 3 to 8 without significant performance cost (tactical moves are rare, keeping branching factor small).


### Enhancement 3: Iterative Deepening

**Motivation:** Fixed-depth search wastes time budget. Iterative deepening maximizes depth within time constraints.

**Implementation:**
1. Complete depth-1 search, save best move
2. Complete depth-2 search, save best move
3. Continue deepening until time budget nearly exhausted
4. Return best move from deepest completed search

**Key feature:** Use evaluation scores from previous depths for evidence-based move ordering in deeper searches.

**Impact:**
- Average depth increased from 3 to 5-7 (sometimes reaching 10 in endgame)
- Adaptive time allocation: more budget for critical mid-game phase (turns 10-40)

---

## Game Context: Freckers

Freckers is a two-player strategy game played on an 8×8 board where players race to move their 6 frogs to the opposite side. Each turn allows:

- **Move** to adjacent lily pad
- **Jump** over opponent frogs (checkers-style)
- **Grow** new lily pads around all frogs

For complete game rules, please refer to **[AI_2025_Project_PartB.pdf](AI_2025_Project_PartB.pdf)**

**Computational Challenges:**
- 20-30 legal moves per position (high branching factor)
- 50-70 turn games (deep game tree)
- 180 second total time limit
- 250MB memory limit

---

## Research: Alternative Approaches

### Monte Carlo Tree Search (MCTS)

We implemented a complete MCTS agent to compare against traditional search:


**Result:** 0% win rate against final Minimax implementation

**Analysis:** For games with clear evaluation criteria and tactical depth, traditional search with good evaluation functions outperformed simulation-based approaches. MCTS shines in domains where evaluation is difficult (e.g., Go), but Freckers' straightforward objectives favor direct evaluation.

### Enhanced Evaluation Functions

Tested several complex evaluation functions:
- Mobility scoring (rewarding more available moves)
- Jump opportunity assessment
- Progressive weighting schemes

**Result:** All performed worse than simple linear evaluation

**Insight:** Complexity doesn't guarantee improvement. The straightforward evaluation capturing core game objectives proved most effective.

### Other Experiments

- **Null-move search:** Improved pruning but reduced accuracy
- **Dynamic evaluation weights:** Added complexity without gains

Complete experimental results in [report.pdf](report.pdf).

---

## Running the Agent

### Prerequisites
- Python 3.12
- Standard library only

### Basic Usage

```bash
# Play agent against itself
python -m referee agent agent

# Test against baseline
python -m referee agent agentgalleries/agent_naive

# Visualization mode 
python -m referee -v 3 -w 0.5 agent agent
```

### Tournament Testing

```bash
# 10-game tournament
python simple_tournament.py agent agentgalleries/agent_naive 10
```

---

## Project Structure

```
COMP30024/
├── agent/                      # Final agent implementation
│   └── program.py             # Minimax + enhancements
├── agentgalleries/            # Alternative implementations
│   ├── agent_naive/           # Baseline heuristic agent
│   ├── agent_mcts/            # MCTS implementation
│   └── agentminiv1/           # Basic minimax (depth 3)
├── referee/                   # Game engine (provided)
├── report.pdf                  # Complete technical report
├── AI_2025_Project_PartB.pdf  # Assignment specification
├── simple_tournament.py       # Tournament benchmarking system
└── team.py
```

---

## Development Methodology

**Iterative Development with Empirical Validation:**

1. Baseline implementation (naive heuristic agent)
2. Core algorithm (basic minimax, depth 3)
3. Enhancement phase (move ordering, quiescence, iterative deepening)
4. Alternative exploration (MCTS implementation and testing)
5. Optimization (code efficiency, time management)

**Testing Protocol:** Every modification validated through tournament testing against previous versions. Changes integrated only if demonstrating measurable improvement in win rate or search efficiency.

**Benchmarking System:** Developed custom tournament runner to automate testing and track performance metrics across agent versions.

---

## Team Contributions

**Yongyou (Lucas) Yu:** Core algorithm implementation (minimax, alpha-beta pruning, move ordering, quiescence search, iterative deepening), evaluation function design, performance optimization, testing and benchmarking

**Zhaoyu (Joey) He:** MCTS exploration, project documentation, report writing

Complete technical details and references available in [report.pdf](report.pdf).


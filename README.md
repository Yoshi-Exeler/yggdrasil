# Simple MiniMax based Chess Engine.

This was a passion project that i used to get familiar with chess engines and improve my programming skills. 
Developement is currently halted until i get motivated to work on this again.
I chose to use a library for board representation and move generation since i was more interested in the actual search code.

Current Performance:
The Engine currently seems to play at roughly 2000 ELO based on matches against other engines with established raings.
The best performing version of the Engine played at ~2200 ELO.

Current Known Problems:
The transposition table incorrectly commits and restores positions. Incorrect assumptions were made while implementing.

Currently Implements the following techniques:
- MiniMax
- Alpha/Beta Pruning
- Quiessence Search
- Standing Pat Optimization
- Small/Big Delta Pruning
- Transposition Table
- Iterative Deepening
- Killer Move Optimization
- Incremental Evaluation
- Lazy SMP Multithreading
- ECO Opening Database
- MVVLVA, PVT, Killer Move based Move Sorting


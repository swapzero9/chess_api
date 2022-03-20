# API for chess engine

This is a repository in which resides an API for my web application. There are several engines in it:

- Random Moves Engine => just because it's funny;
- Monte Carlo Simulation Engine => just a random 2.0;
- MinMax Engine => my proud, but very slow baby;
- 2 Implementations for Artificial Inteligence Engines => unlucky creations I will say;
- Wrapper for Stockfish => so you can compare the rest with the God himself;


In order to run this API, go to subdirectory and run command:
```
uvicorn --reload api.main_api:app
```

In order to install dependencies run command:
```
pip install -r requirements.txt
```

Run and tested on Python 3.9.7

Note: You need to download stockfish engine from [Stockfish site](https://stockfishchess.org/download/ "Stockfish site"), and models for each Ai engine if one wishes to use its full capability.

## Engines directory

In this directory resides every engine in this application. Training of neural networks was done on WSL due to inability to run games in parallel on Windows.
#!/usr/bin/env python3
# bestmove_local.py
import os, shutil
import chess, chess.engine

def find_stockfish():
    """Return path to Stockfish binary or raise if not found."""
    # 1) env var wins
    p = os.environ.get("STOCKFISH_PATH")
    if p and os.path.exists(p):
        return p
    # 2) common Linux paths
    for cand in ("/usr/bin/stockfish", "/usr/games/stockfish"):
        if os.path.exists(cand):
            return cand
    # 3) PATH
    p = shutil.which("stockfish")
    if p:
        return p
    raise FileNotFoundError("Stockfish binary not found. Install it or set STOCKFISH_PATH.")

def best_move_local(fen: str, think_ms: int = 300, options: dict | None = None) -> str:
    """
    Return the best UCI move for the given FEN using a local Stockfish engine.
    - think_ms: time budget in milliseconds
    - options: optional UCI options, e.g. {"Threads": 4, "Skill Level": 10, "Hash": 64}
    """

    engine_path = find_stockfish()
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        if options:
            engine.configure(options)
        result = engine.play(board, chess.engine.Limit(time=think_ms / 1000.0))
        return result.move.uci()

if __name__ == "__main__":
    # Example
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen = "1nbqkbnr/Pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQk - 0 1"
    move = best_move_local(fen, think_ms=300, options={"Threads": 2, "Skill Level": 15})
    print(move)  # e.g., "e2e4"
    

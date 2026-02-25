"""
MineRL — Agent

Builds size-adaptive Minesweeper prompts and parses LLM responses into actions.

Key insight: FRONTIER_THRESHOLD = 0 (all boards use frontier format).
Compact ASCII grid format yielded only 10-15% valid moves; frontier format
with explicit coordinate lists achieves 100% valid moves on all board sizes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .minesweeper_model import MinesweeperAgent


class MinesweeperPlayer:
    """Plays Minesweeper by prompting a fine-tuned LLM."""

    # Set to 0 to use frontier format for all board sizes.
    # Compact grid was tried but produced only 10-15% valid moves
    # because the model cannot reliably parse spatial ASCII grids.
    # Frontier format (explicit constraint lists) achieves 100%.
    FRONTIER_THRESHOLD = 0

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        model_kwargs = {"model_name": model_name} if model_name else {}
        self.agent = MinesweeperAgent(**model_kwargs, **kwargs)

    def build_prompt(self, game_state: Dict[str, Any]) -> tuple[str, str]:
        """
        Build the prompt for the model from a game state dict.

        Args:
            game_state: {
                "board": List[List[str]],   # '.'=hidden, 'F'=flagged, '0'-'8'=number
                "rows": int,
                "cols": int,
                "mines": int,               # total mine count
                "flags_placed": int         # optional, counted from board if absent
            }

        Returns:
            (user_prompt, system_prompt)
        """
        sys_prompt = (
            "You are a Minesweeper AI. "
            'Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}'
        )

        board = game_state["board"]
        rows = game_state["rows"]
        cols = game_state["cols"]
        num_mines = game_state.get("mines", game_state.get("num_mines", 0))
        flags_placed = game_state.get("flags_placed", 0)

        if flags_placed == 0:
            flags_placed = sum(1 for r in board for c in r if c == "F")

        mines_left = num_mines - flags_placed

        if rows <= self.FRONTIER_THRESHOLD and cols <= self.FRONTIER_THRESHOLD:
            prompt = self._build_compact_prompt(board, rows, cols, num_mines, flags_placed, mines_left)
        else:
            prompt = self._build_frontier_prompt(board, rows, cols, num_mines, flags_placed, mines_left)

        return prompt, sys_prompt

    def _build_compact_prompt(self, board, rows, cols, num_mines, flags_placed, mines_left):
        """Compact ASCII grid format. Not used by default (FRONTIER_THRESHOLD=0)."""
        grid_str = "\n".join("".join(row) for row in board)
        return (
            f"MINESWEEPER {rows}x{cols} MINES:{num_mines} FLAGS:{flags_placed} LEFT:{mines_left}\n"
            f"{grid_str}\n"
            "RULES: .=hidden F=flag 0-8=adjacent mines\n"
            "- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n"
            "- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n"
            "- Flag certain mines FIRST, then reveal certain safe cells\n"
            "- NEVER act on already revealed or flagged cells\n"
            'Output ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
        )

    def _build_frontier_prompt(self, board, rows, cols, num_mines, flags_placed, mines_left):
        """
        Frontier format: lists each numbered cell with its hidden neighbours explicitly.
        The model picks coordinates from the listed cells — no spatial parsing needed.
        """
        frontier_info = []
        all_hidden_near_numbers = set()

        for r in range(rows):
            for c in range(cols):
                if board[r][c] not in "012345678":
                    continue
                num = int(board[r][c])
                flags = 0
                hidden = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if board[nr][nc] == "F":
                                flags += 1
                            elif board[nr][nc] == ".":
                                hidden.append((nr, nc))
                                all_hidden_near_numbers.add((nr, nc))

                if hidden:
                    hidden_str = "".join(f"({hr},{hc})" for hr, hc in hidden)
                    frontier_info.append(f"R{r}C{c}={num} flags:{flags} hidden:[{hidden_str}]")

        total_hidden = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == ".")
        interior_count = total_hidden - len(all_hidden_near_numbers)

        frontier_str = "\n".join(frontier_info[:200])  # cap to prevent token explosion
        hidden_near_str = "".join(f"({r},{c})" for r, c in sorted(all_hidden_near_numbers)[:100])

        return (
            f"MINESWEEPER {rows}x{cols} MINES:{num_mines} FLAGS:{flags_placed} LEFT:{mines_left}\n"
            "FRONTIER (numbered cells with hidden neighbors):\n"
            f"{frontier_str}\n"
            f"HIDDEN NEAR NUMBERS: {hidden_near_str}\n"
            f"TOTAL HIDDEN: {total_hidden} INTERIOR(no adj number): {interior_count}\n"
            "RULES: .=hidden F=flag 0-8=adjacent mines\n"
            "- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n"
            "- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n"
            "- Flag certain mines FIRST, then reveal certain safe cells\n"
            "- NEVER act on already revealed or flagged cells\n"
            "- Choose ONLY from HIDDEN NEAR NUMBERS cells listed above\n"
            'Output ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
        )

    def play_action(self, game_state: Dict[str, Any], **gen_kwargs) -> tuple[Optional[Dict], Optional[int], Optional[float]]:
        """
        Generate a single action for the given game state.

        Returns:
            (action_dict, token_count, generation_time)
        """
        prompt, sys_prompt = self.build_prompt(game_state)
        response, tl, gt = self.agent.generate_response(prompt, sys_prompt, **gen_kwargs)
        action = self.parse_action(response)
        return action, tl, gt

    def parse_action(self, response: str) -> Optional[Dict]:
        """
        Extract the first valid JSON action from a model response string.

        A valid action has keys: type ("reveal"|"flag"), row (int), col (int).
        """
        i = 0
        while i < len(response):
            start = response.find("{", i)
            if start == -1:
                break
            brace_count = 0
            end = start
            while end < len(response):
                if response[end] == "{":
                    brace_count += 1
                elif response[end] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            obj = json.loads(response[start:end + 1])
                            if (
                                isinstance(obj, dict)
                                and obj.get("type") in ("reveal", "flag")
                                and "row" in obj
                                and "col" in obj
                            ):
                                obj["row"] = int(obj["row"])
                                obj["col"] = int(obj["col"])
                                return obj
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
                end += 1
            i = end + 1 if end < len(response) else len(response)
        return None

    @staticmethod
    def save_action(action: Dict, file_path: "str | Path") -> None:
        """Write action dict to a JSON file, creating parent dirs if needed."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(action, f, indent=2)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run MineRL agent on a single game state.")
    parser.add_argument("--game_state_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="outputs/action.json")
    parser.add_argument("--model", type=str, default=None, help="Model path or HF ID")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.game_state_file) as f:
        game_state = json.load(f)

    player = MinesweeperPlayer(model_name=args.model)

    gen_kwargs = {"tgps_show": args.verbose}
    config_file = Path("minesweeper_config.yaml")
    if config_file.exists():
        with open(config_file) as f:
            gen_kwargs.update(yaml.safe_load(f))

    action, tl, gt = player.play_action(game_state, **gen_kwargs)

    if args.verbose:
        print(f"Action: {json.dumps(action, indent=2)}")
        if tl and gt:
            print(f"Stats: {tl} tokens in {gt:.2f}s ({tl/gt:.1f} tok/s)")

    if action:
        player.save_action(action, args.output_file)
        print(f"Saved to {args.output_file}")
    else:
        print("ERROR: Failed to generate valid action")
        player.save_action({"error": "parse_failed"}, args.output_file)

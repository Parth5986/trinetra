import sys, ast, inspect

from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)

for mod, fn in [("ppo", "ppo"), ("gae", "calculate_gae"), ("gae", "calculate_returns")]:
    tree = ast.parse(open(rf"{ROOT}\{mod}.py", encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn:
            returns = [
                ast.dump(t) for t in node.body if isinstance(t, ast.Return)
            ]
            args = [a.arg for a in node.args.args]
            print(f"{mod}.{fn} args={args} returns={len(returns)} ret_stmt={returns[0][:60] if returns else 'NONE'}")

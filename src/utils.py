def numbered_string(strings: list[str]):
    return "\n".join(f"[{i+1}]: {s}" for i, s in enumerate(strings))

import os
import pickle
import argparse

from src.engine import Engine
from src.model import Model

DEFAULT_DB_FILE = "engine-loca.pkl" 

def load_engine(filename):
    if os.path.exists(filename):
        print(f"Found engine file: {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Found no engine file: {filename}. Initializing new model and engine...")
        model = Model("all-MiniLM-L6-v2", local_folder_path='./local')

        return Engine(model.encode)


def save_engine(engine, filename):
    with open(filename, "wb") as f:
        pickle.dump(engine, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "engine_file",
        nargs="?",
        default=DEFAULT_DB_FILE,
        help=f"Pickle file to load/save (default: {DEFAULT_DB_FILE})",
    )
    args = parser.parse_args()

    engine_file = args.engine_file
    engine = load_engine(engine_file)

    print("\nCommands:")
    print("  add <text>")
    print("  remove <text>")
    print("  search <query>")
    print("  list")
    print("  exit\n")

    try:
        while True:
            cmd = input(">> ").strip()

            if not cmd:
                continue

            if cmd == "exit":
                break

            parts = cmd.split(" ", 1)
            action = parts[0].lower()

            if action == "add" and len(parts) > 1:
                engine.add([parts[1]]) 
                print(f"Added: '{parts[1]}'")

            elif action == "remove" and len(parts) > 1:
                engine.remove(parts[1])
                print(f"Removed: '{parts[1]}'")

            elif action == "search" and len(parts) > 1:
                results = engine.search(parts[1], top_k=10)
                print(f"Searching for: '{parts[1]}'")
                print("", "score", "match", sep="\t")
                for i, result in enumerate(results):
                    print(
                        f"[{i+1}]",
                        f"{result['score']:.4f}",
                        result["query"],
                        sep="\t",
                    )

            elif action == "list":
                for i, item in enumerate(engine.queries):
                    print(f"[{i+1}]", item, sep="\t")

            else:
                print("Unknown command")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        save_engine(engine, engine_file)


if __name__ == "__main__":
    main()
import os
import pickle
import argparse
from model import Model

DEFAULT_DB_FILE = "local.pkl"


def load_model(filename):
    if os.path.exists(filename):
        print(f"found model file: {filename}")

        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        print(f"found no model file: {filename}")

        return Model(model_name="all-MiniLM-L6-v2")


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_file",
        nargs="?",
        default=DEFAULT_DB_FILE,
        help=f"Pickle file to load/save (default: {DEFAULT_DB_FILE})",
    )
    args = parser.parse_args()

    model_file = args.model_file
    model = load_model(model_file)

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
                model.add(parts[1])

            elif action == "remove" and len(parts) > 1:
                model.remove(parts[1])

            elif action == "search" and len(parts) > 1:
                results = model.search(parts[1], top_k=10)
                print(f"Searching for: '{parts[1]}'")
                print("", "score", "query", sep="\t")
                for i, result in enumerate(results):
                    print(
                        f"[{i+1}]",
                        f"{result['score']:.4f}",
                        result["query"],
                        sep="\t",
                    )

            elif action == "list":
                for i, query in enumerate(model.queries):
                    print(f"[{i+1}]", query, sep="\t")

            else:
                print("Unknown command")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        save_model(model, model_file)


if __name__ == "__main__":
    main()
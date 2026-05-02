import subprocess
import sys


def run(cmd):
    print(f"\n[RUNNING] {cmd}\n")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {cmd}")
        sys.exit(result.returncode)


def main():

    # # 1 Record demonstrations
    run("python -m imitation.record_demos")

    # # # 2 Inspect demos
    run("python -m imitation.inspect_demo")

    # # # 3 Build dataset
    run("python -m imitation.build_dataset")

    # # # 4 Train BC
    run("python -m imitation.train_bc")

    # # # 5 Evaluate model
    run("python -m imitation.evaluate_imitation")

    print("\n✅ Full imitation pipeline finished.")


if __name__ == "__main__":
    main()

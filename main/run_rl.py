import subprocess
import sys


def run(cmd):
    print(f"\n[RUNNING] {cmd}\n")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed: {cmd}")
        sys.exit(result.returncode)


def main():




    # run("python -m rl.train_ppo")


    # run("python -m rl.train_sac")



    print("\n✅ Full rl pipeline finished.")


if __name__ == "__main__":
    main()

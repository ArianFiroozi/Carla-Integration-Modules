import os
import json
import datetime
import subprocess


class ExperimentLogger:

    def __init__(self, experiment_name="experiment", base_dir="experiments"):

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.run_name = f"{timestamp}_{experiment_name}"

        self.base_path = os.path.join(base_dir, self.run_name)

        self.model_dir = os.path.join(self.base_path, "models")
        self.eval_dir = os.path.join(self.base_path, "eval")
        self.logs_dir = os.path.join(self.base_path, "logs")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def save_config(self, config_dict):

        config_path = os.path.join(self.base_path, "config.json")

        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": self.get_git_commit()
        }

        config_dict["metadata"] = metadata

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def log_training(self, step, metrics):

        log_file = os.path.join(self.logs_dir, "train_log.jsonl")

        entry = {
            "step": step,
            "metrics": metrics,
            "time": datetime.datetime.now().isoformat()
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_eval_episode(self, episode_data):

        episode_id = episode_data.get("episode", "unknown")

        file_path = os.path.join(self.eval_dir, f"episode_{episode_id}.json")

        with open(file_path, "w") as f:
            json.dump(episode_data, f, indent=4)

    def get_git_commit(self):

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("ascii").strip()

            return commit

        except:
            return "unknown"

import matplotlib.pyplot as plt
import numpy as np
from .. import config



def plot_discrete_actions(stats):
    """1x2 Subplot for Speed and Turn discrete actions."""
    sp_counts = stats["speed_counts"]
    tr_counts = stats["turn_counts"]
    
    sp_probs = sp_counts / max(1, sp_counts.sum())
    tr_probs = tr_counts / max(1, tr_counts.sum())
    
    sp_labels = [config.SPEED_MAP.get(i, str(i)) for i in range(len(sp_counts))]
    tr_labels = [config.TURN_MAP.get(i, str(i)) for i in range(len(tr_counts))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(sp_labels, sp_probs, color='skyblue', edgecolor='black')
    axes[0].set_title("Speed Action Distribution")
    axes[0].set_ylabel("Probability")
    axes[0].tick_params(axis='x', rotation=30)
    
    axes[1].bar(tr_labels, tr_probs, color='lightgreen', edgecolor='black')
    axes[1].set_title("Turn Action Distribution")
    axes[1].set_ylabel("Probability")
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.show()


def plot_joint_heatmap(joint_counts):
    """Plots a 2D Heatmap of the joint (Speed x Turn) distribution."""
    joint_matrix = np.zeros((5, 4), dtype=np.int64)
    for (speed, turn), count in joint_counts.items():
        if 0 <= speed < 5 and 0 <= turn < 4:
            joint_matrix[speed, turn] = count

    joint_probs = joint_matrix / max(1, joint_matrix.sum())

    plt.figure(figsize=(7, 5))
    im = plt.imshow(joint_probs, cmap="viridis")
    plt.colorbar(im, label="Probability")

    plt.xticks(range(4), [config.TURN_MAP.get(i, str(i)) for i in range(4)])
    plt.yticks(range(5), [config.SPEED_MAP.get(i, str(i)) for i in range(5)])
    plt.xlabel("Turn Action")
    plt.ylabel("Speed Action")
    plt.title("Joint Action Distribution (Speed × Turn)")

    for i in range(5):
        for j in range(4):
            if joint_probs[i, j] > 0:
                plt.text(j, i, f"{joint_probs[i,j]:.2f}", 
                         ha="center", va="center", color="white", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_episode_lengths(episode_lengths):
    """1x1 plot for Episode Lengths."""
    plt.figure(figsize=(6, 4))
    plt.hist(episode_lengths, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title("Episode Lengths Distribution")
    plt.xlabel("Steps per Episode ($T$)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_continuous_deltas(features):
    """1x2 Subplot for Steer Delta and Throttle Delta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    steer_delta = None
    if "steer_delta" in features:
        steer_delta = features["steer_delta"]
    elif "obs_steering_angle" in features:
        steer_delta = np.diff(features["obs_steering_angle"].reshape(-1))

    if steer_delta is not None and len(steer_delta) > 0:
        axes[0].hist(steer_delta, bins=50, color='teal', edgecolor='black', alpha=0.7)
        axes[0].set_title("Steering Angle Delta ($\\Delta$ Steer)")
    else:
        axes[0].set_title("Steering Delta Not Found")

    throttle_delta = None
    if "throttle_delta" in features:
        throttle_delta = features["throttle_delta"]
    elif "obs_throttle" in features:
        throttle_delta = np.diff(features["obs_throttle"].reshape(-1))

    if throttle_delta is not None and len(throttle_delta) > 0:
        axes[1].hist(throttle_delta, bins=50, color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_title("Throttle Delta ($\\Delta$ Throttle)")
    else:
        axes[1].set_title("Throttle Delta Not Found")

    plt.tight_layout()
    plt.show()



def plot_continuous_2d_relationships(features):
    """Plots 2D relationships (hexbin/scatter) for continuous features."""
    pairs_to_plot = [
        ("obs_ego_speed_x", "obs_throttle", "Ego Speed X vs Throttle"),
        ("obs_throttle", "obs_steering_angle", "Throttle vs Steering Angle"),
        ("obs_lane_angle", "obs_steering_angle", "Lane Angle vs Steering Angle"),
        ("obs_ego_in_lane_position_x", "obs_steering_angle", "Lane Offset vs Steering Angle"),
        ("obs_lane_angle", "obs_ego_in_lane_position_x", "Lane Angle vs Lane Offset")
    ]
    
    valid_pairs = [(x, y, title) for (x, y, title) in pairs_to_plot if x in features and y in features]
    if not valid_pairs:
        return

    n = len(valid_pairs)
    cols = 3
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()
    
    for i, (x_key, y_key, title) in enumerate(valid_pairs):
        ax = axes[i]
        x_data = features[x_key]
        y_data = features[y_key]
        
        hb = ax.hexbin(x_data, y_data, gridsize=40, cmap='inferno', mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Count')
        
        ax.set_title(title)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(features):
    """Plots histograms for all collected scalar/array observation features."""
    # Exclude deltas as they are plotted separately
    keys = [k for k in features.keys() if "delta" not in k]
    n = len(keys)
    if n == 0: return
    
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 5, rows * 3))

    for i, k in enumerate(keys):
        plt.subplot(rows, cols, i + 1)
        plt.hist(features[k], bins=50, color='coral', edgecolor='black', alpha=0.7)
        plt.title(k)
        
    plt.tight_layout()
    plt.show()
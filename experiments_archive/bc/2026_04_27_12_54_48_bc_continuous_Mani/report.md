# Experiment Report — 2026_04_27_12_54_48_bc_continuous_Mani
**Creator:** Mani

### Seventeenth experiment:

`experiments_archive\bc\2026_04_27_12_54_48_bc_continuous_Mani`

Added back the 60k data and the model improved. Maybe without those extra spatial features, the model couldn't grasp the whole feature space.
1500 on "without car", 750-800 on "with car".
Also, for a little test to see how it is at turning, I added forced throttle and it got a whopping 1681. Yum.
Let's call this the baseline from now on.

```json
{
  "mode": "continuous",
  "epochs": 30,
  "batch_size": 512,
  "lr": 0.0003,
  "val_split": 0.1,
  "patience": 10,
  "device": "cuda",
  "is_gaussian": false,
  "use_continuous_undersampling": false,
  "undersampling_threshold_continuous": 0.05,
  "undersampling_probability_continuous": 0.6,
  "use_weighted_loss": false,
  "steer_loss_weight_continuous": 3.0,
  "throttle_loss_weight_continuous": 1.0,
  "brake_loss_weight_continuous": 1.0,
  "weighted_loss_threshold_continuous": 0.1,
  "min_std": 0.05,
  "max_std": 1,
  "weight_sampling": "none",
  "cnn_channels": [
    16,
    32,
    64
  ],
  "head_n_mlp_layers": 2,
  "head_mlp_hidden_size": 64,
  "scalar_n_mlp_layers": 2,
  "scalar_mlp_hidden_size": 32,
  "latent_dim": 128,
  "use_one_hot_grid": true,
  "scaling": "z_score",
  "decoupled": false,
  "metadata": {
    "timestamp": "2026-04-27T12:54:48.981823",
    "git_commit": "515614c7e0c20889953ce03b2df026acea0905e9"
  }
}
```


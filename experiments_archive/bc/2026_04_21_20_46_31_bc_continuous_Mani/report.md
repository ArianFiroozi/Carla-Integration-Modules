# Experiment Report — 2026_04_21_20_46_31_bc_continuous_Mani
**Creator:** Mani

### First experiment:

`experiments_archive\bc\2026_04_21_20_46_31_bc_continuous_Mani`

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
  "metadata": {
    "timestamp": "2026-04-21T20:46:31.365238",
    "git_commit": "1ae61521e60bf77d925550e820033434a5fab6bd"
  }
}
```

It did surprisingly well. I'm mostly sure it's because that single flag called `use_one_hot_grid` did the trick, which turns the presence grid into a wall grid, car grid, and ego grid.
Also, the data was from map 1 with no cars (avg 528) and it was tested in the same environment.


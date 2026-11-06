# Experiment Report — 2026_04_23_16_46_39_bc_continuous_Mani
**Creator:** Mani

### Fourteenth experiment:

Fixed a little wrap-around bug in the large angle. Did the baseline + z-score test again on "no car" data and the "no car" environment. Got around 1k (though I think some bad luck was involved).
I'm gonna call this the "baseline with car" from now on.

`experiments_archive\bc\2026_04_23_16_46_39_bc_continuous_Mani`

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
  "metadata": {
    "timestamp": "2026-04-23T16:46:40.000039",
    "git_commit": "0e7adfc5519973d932a6912d6f2a622851fb7dda"
  }
}
```


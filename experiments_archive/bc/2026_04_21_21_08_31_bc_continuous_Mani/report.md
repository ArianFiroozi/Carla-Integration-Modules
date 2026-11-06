# Experiment Report — 2026_04_21_21_08_31_bc_continuous_Mani
**Creator:** Mani

### Third experiment:

Made the network a bit bigger:

```json
  "cnn_channels": [
    32,
    64,
    128
  ],
  "head_n_mlp_layers": 2,
  "head_mlp_hidden_size": 128,
  "scalar_n_mlp_layers": 2,
  "scalar_mlp_hidden_size": 64,
  "latent_dim": 256,
```

Very close performance to the second experiment (avg 300-400ish).


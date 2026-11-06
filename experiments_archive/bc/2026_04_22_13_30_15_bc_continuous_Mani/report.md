# Experiment Report — 2026_04_22_13_30_15_bc_continuous_Mani
**Creator:** Mani

### Fifth and sixth experiment:

I went back to the baseline config. No cars in training or evaluation. So far, normalization was just division by the max value, nothing much.
I added two methods for scaling: z-score and min-max. The driving became way more stable, with z-score going up to a 1k average and min-max up to 1.3k.


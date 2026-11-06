# Experiment Report — 2026_04_23_11_01_11_bc_continuous_Mani
**Creator:** Mani

### Thirteenth experiment:

`experiments_archive\bc\2026_04_23_11_01_11_bc_continuous_Mani`

I'm gonna call this the "baseline without car" from now on.
Went back to the small model, trained on "with car" data (120k data).
Tested on the "no car" environment, did kinda decent (almost 900 compared to 1k-1.2k).
Tested on the "with car" environment and got 600. It acts kinda weird around cars but does have a sense of them being around. (Weirdly, I trained this for another 70 epochs, getting it up to 100 epochs total. I used a patience of 10 epochs to make sure it didn't overfit, but the car mostly forgot how to turn).
(Update: Trained the model again with the wrap-around bug fixed, got similar results).


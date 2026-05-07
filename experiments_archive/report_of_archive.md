Of course! Here is the corrected version of your report. I've fixed grammar and spelling mistakes while keeping your friendly, informal tone.

***

Okay, so first I turned off every kind of flag that might affect the performance. BTW, all eval lengths I'm gonna tell you are an average of 10-20 eval runs (I think the experiments after the fifth one are 20 eval runs each). The runs have random spawn points, so bad luck can happen.

Also, for every experiment I report on, I'll only state the change compared to the previous experiment. If a new kind of config was used, I'll mention the whole config. This was the config for the:

### First experiment:

`experiments\2026_04_21_20_46_31_bc_continuous`

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

### Second experiment:

Switched over to map 1 with cars. It did pretty badly. IDK why, but it seems it did have a sense of other cars' existence. It wasn't good at turning though, unlike previous runs which were better at turning. I mostly suspect the reason is that the dataset is more focused on the existence of other cars (avg 300-400ish).

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

### Fourth experiment:

I mixed the "with car" and "without car" datasets completely and then tested on the "with car" environment. Same results, around 300-400ish (avg).

### Fifth and sixth experiment:

I went back to the baseline config. No cars in training or evaluation. So far, normalization was just division by the max value, nothing much.
I added two methods for scaling: z-score and min-max. The driving became way more stable, with z-score going up to a 1k average and min-max up to 1.3k.

### Seventh and eighth experiment:

I used the "with car" data for training this time and tested on the "no car" environment. Z-score did a better job this time compared to min-max (700 vs 300-400).
It's interesting that when I used the data that has cars in it, for example with z-score, we went from 1k to 700. It seems the car learned to drive faster (because the "with car" dataset was generally about overtaking vehicles or at least matching their speed, which caused the dataset to have a higher average speed). The car also turns the wheel a lot, like it goes near the left wall, then goes to the right wall (that happened quite a few times). Other than that, it seems the model does have a sense of turns and can sometimes turn nicely.
The min-max version, which went from 1.3k to barely 400, was mostly similar to the z-score version, with the only difference being that min-max was less "drunk" and more lazy in turning.

### Ninth and tenth experiment:

I used the models I trained in experiments 7 and 8, but this time the evaluation was with cars. The min-max model had a crazy left bias; it would mostly spawn and immediately turn left, getting very bad results (169 avg). The z-score model did better, even though it crashed a lot, couldn't turn properly near cars, and hit walls. It did have a sense of other cars being around and avoided them pretty nicely a few times. It even did a pretty clean (relatively) overtake once. One reason I think caused that left bias is that the data I recorded with other cars around was a bit more fast-paced, and I usually overtook other cars by going to the left of them and speeding up.

### Eleventh experiment:

`experiments\2026_04_23_09_23_32_bc_continuous`

Went back to the baseline experiment, just with z-score and the bigger model from experiment 3.
It did pretty badly compared to the smaller model (600 vs 1k).

### Twelfth experiment:

Did a little experiment here. I used the 1k baseline and added dataset mirroring. It did worse, around 650, which is weird. It did a mostly similar job, but in two very specific scenarios (like two specific places on the map), it showed some super crazy, random behavior. I think that was mostly the reason it did worse by quite a lot (also, there is always some luck involved).

### Thirteenth experiment:

`experiments\2026_04_23_11_01_11_bc_continuous`

I'm gonna call this the "baseline without car" from now on.
Went back to the small model, trained on "with car" data (120k data).
Tested on the "no car" environment, did kinda decent (almost 900 compared to 1k-1.2k).
Tested on the "with car" environment and got 600. It acts kinda weird around cars but does have a sense of them being around. (Weirdly, I trained this for another 70 epochs, getting it up to 100 epochs total. I used a patience of 10 epochs to make sure it didn't overfit, but the car mostly forgot how to turn).
(Update: Trained the model again with the wrap-around bug fixed, got similar results).

### Fourteenth experiment:

Fixed a little wrap-around bug in the large angle. Did the baseline + z-score test again on "no car" data and the "no car" environment. Got around 1k (though I think some bad luck was involved).
I'm gonna call this the "baseline with car" from now on.

`C:\carla\Carla-Integration-Modules\experiments\2026_04_23_16_46_39_bc_continuous`

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

### Fifteenth experiment:

`experiments\2026_04_23_18_04_25_bc_continuous`

Same experiment as the thirteenth but I drove some more. Got worse results: 500 on the "no car" environment (pretty bad driving compared to the smaller dataset), and 338 on the "with car" environment (really bad).
Like, it low-key reacts to cars, but doesn't really drive well. Weirdly, when I mixed newer data (60k) with a portion of older data (40k), it resulted in a performance similar to before I added the extra 60k. I'm gonna put that 60k away for now.

### Sixteenth experiment:

`experiments\2026_04_27_12_16_57_bc_continuous`

Added 4 spatial features, scalars to detect the nearest walls and vehicles.
Adding this to the "baseline with car," we got about 500 for the "with car" environment and again about 1k on the "no car" environment.

From this experiment onward, we're gonna remove the forced throttle and just give throttle a little boost (if throttle > 0.05 and < 0.13, we bump it up to 0.13).

### Seventeenth experiment:

`experiments\2026_04_27_12_54_48_bc_continuous`

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

### Eighteenth experiment:

`experiments\2026_04_27_13_45_04_bc_continuous`

Decoupled the head. Got about 800 on "with car" and 1300 on "without car". Can't really say if it did better than the coupled version or worse; not enough evidence for either. It was okay, I guess.

### Nineteenth experiment:

`experiments\2026_04_27_22_21_40_bc_continuous`

Used the baseline with `is_gaussian` set to true. It did pretty badly. Like, it had a sense of turning and other cars, but it looked lazy. Instead of turning normally, it very slowly rotated the wheel, causing it to crash on a lot of turns. (I only ran this model on the "with car" environment, not the "no car" one).

### Twentieth experiment:

`experiments\2026_04_27_22_33_48_bc_continuous`

Same as the previous experiment, just decoupled the head. It did a bit better but was still very bad compared to the non-Gaussian version.

### Twenty-first and second experiment:

`experiments\2026_04_28_12_13_00_bc_continuous`
`experiments\2026_04_28_12_36_57_bc_continuous`

Baseline, with a Gaussian head and the bigger network. Nope. Although, it did slightly better than the small network version of this experiment (experiment nineteen).
Also increased the `min_std` a bit again, still kinda meh, although it did do better than the 0.05 `min_std`.
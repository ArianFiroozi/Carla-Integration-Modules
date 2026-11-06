# Experiment Report — 2026_04_22_14_10_14_bc_continuous_Mani
**Creator:** Mani

### Seventh and eighth experiment:

I used the "with car" data for training this time and tested on the "no car" environment. Z-score did a better job this time compared to min-max (700 vs 300-400).
It's interesting that when I used the data that has cars in it, for example with z-score, we went from 1k to 700. It seems the car learned to drive faster (because the "with car" dataset was generally about overtaking vehicles or at least matching their speed, which caused the dataset to have a higher average speed). The car also turns the wheel a lot, like it goes near the left wall, then goes to the right wall (that happened quite a few times). Other than that, it seems the model does have a sense of turns and can sometimes turn nicely.
The min-max version, which went from 1.3k to barely 400, was mostly similar to the z-score version, with the only difference being that min-max was less "drunk" and more lazy in turning.


### Ninth and tenth experiment:

I used the models I trained in experiments 7 and 8, but this time the evaluation was with cars. The min-max model had a crazy left bias; it would mostly spawn and immediately turn left, getting very bad results (169 avg). The z-score model did better, even though it crashed a lot, couldn't turn properly near cars, and hit walls. It did have a sense of other cars being around and avoided them pretty nicely a few times. It even did a pretty clean (relatively) overtake once. One reason I think caused that left bias is that the data I recorded with other cars around was a bit more fast-paced, and I usually overtook other cars by going to the left of them and speeding up.


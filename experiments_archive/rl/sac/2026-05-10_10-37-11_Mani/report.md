# Experiment Report — 2026-05-10_10-37-11_Mani
**Creator:** Mani

eighted experiment:
experiments_archive\rl\sac\2026-05-10_10-37-11_Mani

maybe just maybe that action proccesing inside the env is causing the issue, removed it ,  kept only crit warmup up until 40k, all same issues as soon as acotr Starts learning everything goes bad
one intersting difference with first experiment(which the only difference they have is basically action proccesing) is here the actor loss starts going up immediately while in that experiment actor_loss goes down for a while
then moves up , this could be due to fact that the car is not moving at all so we keep seeing the same obs, i think even small brakes causes throttle to not work so  wihtout Brake/throttle exclusivity car isnt
going to move at all.


# Experiment Report — 2026-05-13_19-45-23_Mani
**Creator:** Mani

fourtheeth experiment:
experiments_archive\rl\sac\2026-05-13_19-45-23_Mani
same config as twleth experiment but halved actor lr ( right now is: ACTOR_LR = 5e-7 CRITIC_LR = 1e-4          
ALPHA_LR = 1e-4) also changed actor and alpha update to 1 every 20 critic update(crtitc updates every step) ,  results were like this:
at 100k when actor unfroze the return and length of episodes collapsed(but a lesser collapse compared to twelth, like in twelth ) but it mostly stay there only very briefly it went up a little(like reaching 30% of bc) around 160k-180
but then went back where it was(low but not compeletely destryoed like 15% of bc , unlike twelth experiment where collapses  were utterly bad ) , it ran until 250k


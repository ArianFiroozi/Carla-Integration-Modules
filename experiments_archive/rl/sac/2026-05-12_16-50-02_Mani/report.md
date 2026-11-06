# Experiment Report — 2026-05-12_16-50-02_Mani
**Creator:** Mani

twelth experiment:
experiments_archive\rl\sac\2026-05-12_16-50-02_Mani
changed train step per update , set it to 10  for actor and alpha and 1 for crtitc. and lowered lr even more down to 1e-6.
what happending is the moment acotr got unfrozen , everything went nuts, alpha itself and crit loss went up rewards and length collapsed
action mean brake throotle went up like crazy (steer kinda did a full sin like wave) entropy went up q means went up std went up , after 150k things started to go back (basically everything started cooling down) and even length and reward went up by 200k and still i amt raining
it feels like network unlearned stuff and now is learning again. after a while it reconvered and actually kinda  beat bc once it hit 280-320k but then right now it is at 350k and it basically collapsed again , every symptom that we saw before kinda happened again.


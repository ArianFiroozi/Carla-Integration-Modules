# Experiment Report — 2026-05-25_11-56-28_Mani
**Creator:** Mani

sixteeth experiment:
experiments_archive\rl\sac\2026-05-22_19-20-56_Mani
experiments_archive\rl\sac\2026-05-22_21-30-20_Mani
experiments_archive\rl\sac\2026-05-25_11-56-28_Mani
experiments_archive\rl\sac\2026-05-25_13-30-07_Mani
added q normalzation and bc penalty (tested beta = 10 and 1000) all still the same issue , it seems 100k crtitc is needed since 12th adn 14th experiment where crtitc was learning until 100k alone were the only experiments
where grad norm of critic didnt explode. same problems grad of crtitc explodes, i think the -200 causes explosions because we bacially repeated 12 and 14 experiment 
but the results were bad .


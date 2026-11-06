# Experiment Report — 2026-05-09_13-48-45_Mani
**Creator:** Mani

fifth experiment:
experiments_archive\rl\sac\2026-05-09_13-48-45_Mani
ran with only the critic training(160k)
very noising episode return and length , crtitc loss was most of the time near 0.02 but with a lot of jumps spikes to 1.0 id say like 10 jump spikes at least across each 10k episodes
now some intersting stuff , replay/reward_std around 0.3 to 0.4 with like 10-20 jump spikes to 0.8 every 20k steps with crtitc 1 and 2 std also having a very very similiar pattern , replay min and crtitcs min also looking simliar with usually being around
0- -1 but dropping to 8 and coming back (spikes) ,   crtitcs q mean and replay/reward_mean also being really similiar in shpae but replay moving in 0.5 to 0.6 range while critics moving in 0.3-0.4 range, crtitc difference slowly going down but in an usttable way
(from 0 to 100k) and kinda looks like they arent going down  from 100k - 160k , i think crtitc can learn when on its own


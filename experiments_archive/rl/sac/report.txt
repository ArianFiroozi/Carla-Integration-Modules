First experiment:

experiments_archive\rl\sac\2026-05-07_21-22-18_Mani


based on plots : 

*   **`critic/q1_max`**: The line is flat at $0$ until approximately $40$k steps, then it moves steadily upwards, reaching roughly $60$ by $200$k steps.
*   **`critic/q1_mean`**: The line is flat near $0$ until ~$40$k steps, then moves sharply up to peak around $10$-$12$ between $75$k and $100$k steps. After $100$k, it becomes very noisy and generally trends downward back toward $0$-$5$ by $200$k.
*   **`critic/q1_min`**: Flat near $0$ until ~$40$k steps, then it becomes extremely noisy with severe downward spikes reaching $-60$ or lower for the remainder of the plot up to $200$k.
*   **`critic/q1_std`**: Flat near $0$ until ~$40$k steps, then moves steadily upwards in a curve, reaching about $25$-$30$ by $200$k with moderate noise.
*   **`critic/q2_mean`**: Shows a nearly identical pattern to `q1_mean`. Flat until ~$40$k, spikes up to ~$10$-$12$ around $80$k-$100$k, then becomes noisy and moves downward toward $0$-$5$ by $200$k.
*   **`critic/q2_std`**: Shows a nearly identical pattern to `q1_std`. Flat until ~$40$k, then steadily moves upward to roughly $25$ by $200$k.
*   **`critic/q_diff_mean`**: Starts around $0.015$, moves down to a low of ~$0.002$ around $30$k-$40$k, moves sharply back up to ~$0.02$ by $75$k, and then stays noisy but relatively horizontal between $0.015$ and $0.025$ until $200$k.


*   **`train/actor_loss`**: Flat at $0$ until ~$40$k, then moves sharply downward to a trough of roughly $-12$ around $80$k-$100$k. From $100$k to $200$k, it trends back upward toward $0$ while becoming increasingly noisy.
*   **`train/alpha`**: Flat at $0.02$ until ~$40$k, dips down to $0.015$, spikes up to a peak of ~$0.032$ at ~$70$k, drops and spikes again to ~$0.027$ at ~$110$k, and then generally trends downward to $0.015$ by $200$k.
*   **`train/alpha_loss`**: Flat at $0$ until ~$40$k, then immediately becomes extremely noisy, oscillating wildly between $-3$ and $2+$ with no meaningful upward or downward trend for the rest of the plot.
*   **`train/critic_loss`**: Flat near $0$ until ~$40$k, then becomes very noisy, characterized by frequent, extreme upward spikes (often hitting $10$ to $15+$) all the way to $200$k.
*   **`train/episode_length`**: Highly noisy from the very beginning. It violently oscillates between $0$ and $2000$ throughout the entire $200$k steps, frequently hitting the $2000$ ceiling, especially after $100$k.
*   **`train/episode_return`**: Starts high and noisy (fluctuating between $0$ and $600$) until ~$50$k, drops to around $0$, and then exhibits a noisy but clear downward trend, reaching roughly $-200$ to $-600$ by $200$k.


*   **`replay/buffer_size`**: A perfect, straight diagonal line moving constantly upward from $0$ at step $0$ to roughly $200$k at step $200$k.
*   **`replay/reward_max`**: Noisy and fluctuates mostly between $0.9$ and $1.05$ across the entire $200$k steps, with a very slight overall downward drift but mostly just noise.
*   **`replay/reward_mean`**: Exhibits a clear, steady downward trend from roughly $0.6$ at the start down to nearly $0$ by $200$k steps, with moderate noise along the way.
*   **`replay/reward_min`**: Mostly sits flat near $0$, but has constant, extreme downward spikes hitting $-10$ repeatedly throughout the entire $200$k steps.
*   **`replay/reward_std`**: Highly volatile and noisy from start to finish, rapidly jumping back and forth between roughly $0.3$ and $1.0$ across the entire $200$k steps.



i ran the best model  and watched , the model was slightlyyyyy better at not hitting other cars from side but was kinda hestinatn to braken when getting near them from behind 

and for latest checkpoint it was just awful , it mostly sat there not moving and when it started moving it was random steering to left, and those times it was stationary rewards where all near each other in a -100 to 100 range


second experiment:

found a bug in first experiment reward , fixed it , same results



third experiment:


turned off critic warm up to see what happens , after 50k steps model doenst want to move at all , forgot basic driving skills 



fourth experiment:
experiments_archive\rl\sac\2026-05-09_12-18-41_Mani
lowered init var and max var ,  ran for 50k again , very simliar results to both driving skills and plot 
one intersting thing is q values plots where bad , mean was consitently going down and var was going up



fifth experiment:
experiments_archive\rl\sac\2026-05-09_13-48-45_Mani
ran with only the critic training(160k)
very noising episode return and length , crtitc loss was most of the time near 0.02 but with a lot of jumps spikes to 1.0 id say like 10 jump spikes at least across each 10k episodes
now some intersting stuff , replay/reward_std around 0.3 to 0.4 with like 10-20 jump spikes to 0.8 every 20k steps with crtitc 1 and 2 std also having a very very similiar pattern , replay min and crtitcs min also looking simliar with usually being around
0- -1 but dropping to 8 and coming back (spikes) ,   crtitcs q mean and replay/reward_mean also being really similiar in shpae but replay moving in 0.5 to 0.6 range while critics moving in 0.3-0.4 range, crtitc difference slowly going down but in an usttable way
(from 0 to 100k) and kinda looks like they arent going down  from 100k - 160k , i think crtitc can learn when on its own





sixth experiment:

ran this for 80k: removed crtitc warmp up , froze the log std head , loss went up even worse than before and q mean went down like crazy , alpha loss was the only thing that went down kinda noramlly(and alpha itself went up a lot , didnt explode but clearly went up )


seventh experiment:
experiments_archive\rl\sac\2026-05-10_09-27-46_Mani
same as sixth experiment but forze aplha from updating too, after just 30k steps policy is crashing very early(low episode length) and rewards went from moving from -50 to 500 to like moving wihtitn -20 +10 
very fast.




eighted experiment:
experiments_archive\rl\sac\2026-05-10_10-37-11_Mani

maybe just maybe that action proccesing inside the env is causing the issue, removed it ,  kept only crit warmup up until 40k, all same issues as soon as acotr Starts learning everything goes bad
one intersting difference with first experiment(which the only difference they have is basically action proccesing) is here the actor loss starts going up immediately while in that experiment actor_loss goes down for a while
then moves up , this could be due to fact that the car is not moving at all so we keep seeing the same obs, i think even small brakes causes throttle to not work so  wihtout Brake/throttle exclusivity car isnt
going to move at all. 


nineth experiment:
experiments_archive\rl\sac\2026-05-10_13-45-26_Mani
thought maybe the problem is alpha so forced it to zero, simliar to previous runs the moment actor starts learning the rewards and length crash to almost zero.




tenth and eleveth experiment:


lowered actor lr(from 1e-4 to 1e-5) , removed weight decay, no alpha , both warm up on and off the same issue happens as always.







twelth experiment:
experiments_archive\rl\sac\2026-05-12_16-50-02_Mani
changed train step per update , set it to 10  for actor and alpha and 1 for crtitc. and lowered lr even more down to 1e-6.
what happending is the moment acotr got unfrozen , everything went nuts, alpha itself and crit loss went up rewards and length collapsed
action mean brake throotle went up like crazy (steer kinda did a full sin like wave) entropy went up q means went up std went up , after 150k things started to go back (basically everything started cooling down) and even length and reward went up by 200k and still i amt raining
it feels like network unlearned stuff and now is learning again. after a while it reconvered and actually kinda  beat bc once it hit 280-320k but then right now it is at 350k and it basically collapsed again , every symptom that we saw before kinda happened again.


thirtheeth experiment: 
reduced warm up from 100k to 30k(cuz in most experiments above crit was stable by the time it hit 30k steps) , simmliar thing to previous experiment happened but the re learn (before collapsed) was worse so like it started unlearning stuff from 30k to 120k tne re learning a little bit
from 120k to 170k but then staeted to go down again , i stopped here.



fourtheeth experiment:
experiments_archive\rl\sac\2026-05-13_19-45-23_Mani
same config as twleth experiment but halved actor lr ( right now is: ACTOR_LR = 5e-7 CRITIC_LR = 1e-4          
ALPHA_LR = 1e-4) also changed actor and alpha update to 1 every 20 critic update(crtitc updates every step) ,  results were like this:
at 100k when actor unfroze the return and length of episodes collapsed(but a lesser collapse compared to twelth, like in twelth ) but it mostly stay there only very briefly it went up a little(like reaching 30% of bc) around 160k-180
but then went back where it was(low but not compeletely destryoed like 15% of bc , unlike twelth experiment where collapses  were utterly bad ) , it ran until 250k




fifteeth experiment:
experiments_archive\rl\sac\2026-05-22_11-14-10_Mani
simliar to thirtheeth experiment with the difference that collision penalty was increased to -500


sixteeth experiment:
experiments_archive\rl\sac\2026-05-22_19-20-56_Mani
experiments_archive\rl\sac\2026-05-22_21-30-20_Mani
experiments_archive\rl\sac\2026-05-25_11-56-28_Mani
experiments_archive\rl\sac\2026-05-25_13-30-07_Mani
added q normalzation and bc penalty (tested beta = 10 and 1000) all still the same issue , it seems 100k crtitc is needed since 12th adn 14th experiment where crtitc was learning until 100k alone were the only experiments
where grad norm of critic didnt explode. same problems grad of crtitc explodes, i think the -200 causes explosions because we bacially repeated 12 and 14 experiment 
but the results were bad .


seventheeth experiment:
experiments_archive\rl\sac\2026-05-30_18-23-26_Mani

added replay buffer mixing(50% expert data 50% sac collected) but removed bc penalty ,same problem as always.

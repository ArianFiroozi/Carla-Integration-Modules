# Experiment Report — 2026-05-07_21-22-18_Mani
**Creator:** Mani

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


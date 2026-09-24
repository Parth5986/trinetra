def calculate_gae(rewards, values, dones, lam, gamma, terminals=None):
    """Generalized Advantage Estimation with terminal/truncation awareness.

    - terminal step (env terminated)          -> bootstrap with 0, GAE chain stops
    - truncated step (hit max_steps)          -> bootstrap with the last value so
                                                 long episodes keep their tail credit
    - normal step                             -> bootstrap with values[t+1]

    `terminals` is a per-step bool list (True = env came to a true end, False =
    truncated or mid-episode). When it is None the old behaviour is kept
    (every done is treated as terminal).
    """
    advantages = []
    gae = 0.0

    last = len(rewards) - 1

    for t in range(last, -1, -1):

        if terminals is not None and dones[t]:
            # Episode boundary. Distinguish a genuine terminal from a
            # truncation so max_steps endings don't get under-credited.
            if terminals[t]:
                next_value = 0.0
                cont = 0.0
            else:
                next_value = values[t]
                cont = 1.0
        else:
            next_value = 0.0 if t == last else values[t + 1]
            cont = 0.0 if t == last else 1.0

        td_error = (
            rewards[t]
            + gamma * next_value * cont
            - values[t]
        )

        gae = (
            td_error
            + gamma * lam * cont * gae
        )

        advantages.append(gae)

    advantages.reverse()

    return advantages


def calculate_returns(advantages, values):
    returns = []

    for i in range(len(advantages)):
        returns.append(
            advantages[i] + values[i]
        )

    return returns
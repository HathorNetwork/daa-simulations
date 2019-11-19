from collections import defaultdict
from math import log


def sum_weights(w1: float, w2: float) -> float:
    return aux_calc_weight(w1, w2, 1)


def sub_weights(w1: float, w2: float) -> float:
    if w1 == w2:
        return 0
    return aux_calc_weight(w1, w2, -1)


def aux_calc_weight(w1: float, w2: float, multiplier: int) -> float:
    a = max(w1, w2)
    b = min(w1, w2)
    if b == 0:
        # Zero is a special acc_weight.
        # We could use float('-inf'), but it is not serializable.
        return a
    return a + log(1 + 2**(b - a) * multiplier, 2)


def get_orphans(miner, *, min_timestamp: float = 0):
    seen = set(x.hash for x in miner.best_block.get_blockchain())

    for block in miner.get_blocks():
        if block.hash not in seen:
            if block.timestamp > min_timestamp:
                yield block


def blocks_per_miner(miner):
    d = defaultdict(int)
    for block in miner.get_blocks():
        d[block.miner] += 1
    return d


def _plot_difficulty(manager, *, min_timestamp=0):
    miner = manager.miners[0]
    x_factor = 3600 * 24   # days

    import matplotlib.pyplot as plt
    x_values = []
    y_values = []
    for block in miner.best_block.get_blockchain():
        if block.timestamp < min_timestamp:
            continue
        x_values.append(block.timestamp / x_factor)
        y_values.append(block.weight - log(manager.target, 2))
    plt.scatter(x_values, y_values, marker='.')

    v = []
    for timestamp, hashrate in sorted(manager.total_hashrate_history.items()):
        if v:
            v.append((timestamp - 1, v[-1][1]))
        v.append((timestamp, hashrate))
    v.append((manager.seconds(), v[-1][1]))
    x_values = [x / x_factor for x, y in v]
    y_values = [log(max(y, 1), 2) for x, y in v]
    plt.plot(x_values, y_values, 'k')


def _plot_number_of_blocks(manager):
    import matplotlib.pyplot as plt
    miner = manager.miners[0]
    x_factor = 3600 * 24   # days
    x_values = [block.timestamp / x_factor for block in miner.best_block.get_blockchain()]
    y_values = list(len(x_values) - x for x in range(len(x_values)))
    plt.plot(x_values, y_values, '-', linewidth=0.5)
    plt.plot([0, manager.seconds() / x_factor], [0, manager.seconds() / manager.target], 'k')


def plot_difficulty(managers, *, min_timestamp=0, save_to=None):
    import matplotlib.pyplot as plt

    print('Total: {} loops'.format(len(managers)))

    title = ', '.join(list(set(x.daa.__class__.__name__ for x in managers)))

    grid = plt.GridSpec(4, 1, wspace=0.4, hspace=0.3)
    plt.subplot(grid[:3, 0])
    plt.xlabel('Time (days)')
    plt.ylabel('Weight')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.minorticks_on()
    plt.grid(which='both', linewidth=0.25, linestyle='--')
    for manager in managers:
        _plot_difficulty(manager, min_timestamp=min_timestamp)

    plt.subplot(grid[3, 0])
    plt.ylabel('Blocks')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.minorticks_on()
    plt.grid(which='both', linewidth=0.25, linestyle='--')
    for manager in managers:
        _plot_number_of_blocks(manager)

    if save_to:
        plt.savefig(save_to, dpi=300)
    else:
        plt.show()

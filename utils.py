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


def all_blocks_per_miner(miner):
    d = defaultdict(int)
    for block in miner.get_blocks():
        d[block.miner] += 1
    return d


def best_blockchain_per_miner(manager):
    miner = manager.miners[0]
    d = defaultdict(int)
    for block in miner.best_block.get_blockchain():
        d[block.miner] += 1
    return d


def blocks_per_miner(it):
    d = defaultdict(int)
    for block in it:
        d[block.miner] += 1
    return d


def _plot_difficulty(manager, *, min_timestamp=0, highlight_miner=None):
    miner = manager.miners[0]
    x_factor = 3600 * 24   # days

    import matplotlib.pyplot as plt
    x_values = []
    y_values = []
    x2_values = []
    y2_values = []
    for block in miner.best_block.get_blockchain():
        if block.timestamp < min_timestamp:
            continue
        if block.miner == highlight_miner:
            x2_values.append(block.timestamp / x_factor)
            y2_values.append(block.weight - log(manager.target, 2))
        else:
            x_values.append(block.timestamp / x_factor)
            y_values.append(block.weight - log(manager.target, 2))
    plt.scatter(x_values, y_values, marker='.')
    if x2_values:
        plt.scatter(x2_values, y2_values, marker='x')

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


def plot_difficulty(managers, *, min_timestamp=0, save_to=None, highlight_miner=None):
    import matplotlib.pyplot as plt

    print('Total: {} loops'.format(len(managers)))

    title = ', '.join(list(set(x.daa.__class__.__name__ for x in managers)))

    grid = plt.GridSpec(4, 1, wspace=0.4, hspace=0.3)
    plt.subplot(grid[:3, 0])
    plt.xlabel('Time (days)')
    plt.ylabel('Difficulty ($log_2(H)$)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(title)
    plt.minorticks_on()
    plt.grid(which='both', linewidth=0.25, linestyle='--')
    for manager in managers:
        _plot_difficulty(manager, min_timestamp=min_timestamp, highlight_miner=highlight_miner)

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


def mining_stats(managers, save_to=None):
    for manager in managers:
        blocks = best_blockchain_per_miner(manager)
        hashes = dict((x.name, x.hashrate) for x in manager.miners)
        miners = dict((x.name, x) for x in manager.miners)

        total_blocks = sum(x for x in blocks.values())
        total_hash = sum(x for x in hashes.values())
        blocks_percent = dict((k, v / total_blocks) for k, v in blocks.items())
        hashes_percent = dict((k, v / total_hash) for k, v in hashes.items())

        names = [x.name for x in manager.miners]
        names.sort()

        for x in names:
            miner = miners[x]
            print('{}'.format(x))
            print('  log(H) = {:.6f} frac={:.6f}'.format(
                log(miner.hashrate, 2),
                hashes_percent.get(x, 0),
            ))
            print('  best block height={} weight={} logwork={}'.format(
                miner.best_block.height,
                miner.best_block.weight,
                miner.best_block.logwork,
            ))
            print('  {} blocks ({:.6f})'.format(
                blocks.get(x, 0),
                blocks_percent.get(x, 0),
            ))
            total_orphans = len(list(get_orphans(miner)))
            print('  {} orphan blocks'.format(total_orphans))
            print('  orphan', blocks_per_miner(get_orphans(miner)).items())
            print('  all_blocks', blocks_per_miner(miner.known_blocks.values()).items())
            print('')

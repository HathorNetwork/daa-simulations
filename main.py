import argparse
import os
import pickle
import sys
from math import ceil, inf

import utils
from difficulty import CRAZY, HTR, LWMA, MSB
from simulator import EventType, Manager, Miner, Miner51Attack, MinerDoubleAvgTimestamp

sys.setrecursionlimit(10000)

def run_flat(manager):
    manager.start()
    manager.addMiner(Miner(15 * 2**40, is_quiet=True))
    manager.addMiner(Miner(15 * 2**40, is_quiet=True))
    manager.addMiner(Miner(30 * 2**40, is_quiet=True))
    manager.addMiner(Miner(30 * 2**40, is_quiet=True))
    manager.addMiner(Miner(45 * 2**40, is_quiet=True))
    manager.run(3600 * 24 * 5, show_progress=True)
    return manager

def run_ts_tampering(manager):
    manager.start()
    manager.addMiner(Miner(25 * 2**40, is_quiet=True))
    manager.addMiner(Miner(30 * 2**40, is_quiet=True))
    manager.addMiner(Miner(45 * 2**40, is_quiet=True))
    manager.run(3600 * 24, show_progress=True)
    manager.addMiner(MinerDoubleAvgTimestamp(10 * 2**40, is_quiet=True))
    manager.run(3600 * 24 * 4, show_progress=True)
    return manager

def run_51attack_90(manager, p=0.9):
    manager.start()
    manager.addMiner(Miner(15 * 2**40, is_quiet=True))
    manager.addMiner(Miner(30 * 2**40, is_quiet=True))
    manager.addMiner(Miner(45 * 2**40, is_quiet=True))

    total_hashrate = sum(x.hashrate for x in manager.miners)
    attacker = Miner51Attack(int(total_hashrate * p / (1 - p)), is_quiet=True)
    manager.addMiner(attacker)
    attacker.stop()

    manager.run(3600 * 24 * 1, show_progress=True)
    attacker.start()
    attacker.start_attack(manager.genesis)
    manager.run(3600 * 24 * 1, show_progress=True)
    attacker.stop_attack()
    attacker.stop()
    manager.run(3600 * 24 * 1, show_progress=True)

    return manager

def run_51attack_10(manager):
    return run_51attack_90(manager, p=0.1)

def run_51attack_50(manager):
    return run_51attack_90(manager, p=0.50)

def run_4x(manager):
    manager.start()
    manager.addMiner(Miner(25 * 2**20, is_quiet=True))
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.addMiner(Miner(100 * 2**20, is_quiet=True))
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.stopMiner(1)
    manager.run(3600 * 24 * 2, show_progress=True)
    return manager

def _run1(manager, hashrate, multiplier):
    manager.start()
    manager.addMiner(Miner(hashrate, is_quiet=True))
    manager.addMiner(Miner(multiplier * hashrate, is_quiet=True))
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.stopMiner(1)
    if multiplier > 10000:
        manager.run(3600 * 24 * 400, until_ev_type=EventType.NEW_BLOCK, show_progress=True)
    manager.run(3600 * 24 * 4, show_progress=True)
    return manager

def run_1k(manager):
    return _run1(manager, 25 * 2**20, 1000)

def run_1M(manager):
    return _run1(manager, 25 * 2**20, 1000000)

def run_w60(manager):
    return _run1(manager, 2**50, 1024)

def run_w70(manager):
    return _run1(manager, 2**60, 1024)

def run_preview20x(manager):
    manager.start()
    manager.addMiner(Miner(630 * (2**20), is_quiet=True))
    manager.addMiner(Miner(630 * (2**20), is_quiet=True))
    manager.run(3600 * 1.5, show_progress=True)
    manager.addMiner(Miner(18 * (2**30), is_quiet=True))
    manager.run(3600 * 1, show_progress=True)
    manager.stopMiner(2)
    manager.run(3600 * 6, show_progress=True)
    return manager

def run_preview1000x(manager):
    manager.start()
    manager.addMiner(Miner(630 * (2**20), is_quiet=True))
    manager.addMiner(Miner(630 * (2**20), is_quiet=True))
    manager.run(3600 * 3, show_progress=True)
    manager.addMiner(Miner(18 * (2**30), is_quiet=True))
    manager.run(3600 * 1, show_progress=True)
    manager.stopMiner(0)
    manager.stopMiner(1)
    manager.stopMiner(2)
    manager.addMiner(Miner(2**24, is_quiet=True))
    manager.run(3600 * 12, show_progress=True)
    return manager

daa_choices = {
    'lwma': lambda: LWMA(n=134, tl_rules=False),
    'htr': lambda: HTR(),
    'htr134': lambda: HTR(n=134, max_dw_rule=False),
    'crazy': lambda: CRAZY(n=134, max_dw_rule=False),
    'msb': lambda: MSB(n=134 * 2),
}

profile_choices = dict((f.__name__, f) for f in [
    run_1k,
    run_1M,
    run_4x,
    run_w60,
    run_w70,
    run_flat,
    run_51attack_90,
    run_51attack_50,
    run_51attack_10,
    run_ts_tampering,
    run_preview20x,
    run_preview1000x,
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('daa', help='DAA algorithm', choices=list(daa_choices.keys()))
    parser.add_argument('profile', help='Profile', choices=list(profile_choices.keys()))
    parser.add_argument('n', type=int, help='number of loops')
    parser.add_argument('--reset', action='store_true', help='overwrite previous results')
    parser.add_argument('--weight-decay', action='store_true', help='allow weight decay')
    parser.add_argument('--solo', action='store_true', help='run and skip saving')
    parser.add_argument('--highlight-miner', help='miner to be highlighted')
    args = parser.parse_args()

    daa = daa_choices[args.daa]()
    profile = profile_choices[args.profile]

    print('Running', args)

    managers = []
    for i in range(args.n):
        print('Loop {}'.format(i + 1))
        manager = Manager(daa=daa, weight_decay=args.weight_decay)
        managers.append(profile(manager))

    utils.mining_stats(managers)

    suffix = ''
    if args.weight_decay:
        suffix = '-decay'
    basename = './data/daa-{}-{}{}'.format(args.profile, args.daa, suffix)
    if not args.solo:
        filename = '{}.pickle'.format(basename)
        if not args.reset and os.path.isfile(filename):
            previous = pickle.load(open(filename, 'rb'))
            managers = previous + managers

        fp = open(filename, 'wb')
        pickle.dump(managers, fp)
        fp.close()
        print('Saved to {} (total={} loops)'.format(filename, len(managers)))

    img_filename = '{}.png'.format(basename)
    utils.plot_difficulty(managers, save_to=img_filename, highlight_miner=args.highlight_miner)
    os.system('open {}'.format(img_filename))


if __name__ == '__main__':
    main()

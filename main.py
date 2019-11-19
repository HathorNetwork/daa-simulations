from simulator import Manager, Miner, EventType
from difficulty import HTR, LWMA, MSB, CRAZY
from math import inf, ceil
import os
import pickle
import argparse
import utils

def run_4x(daa):
    manager = Manager(daa=daa)
    manager.start()
    manager.addMiner(Miner(25 * 2**20, is_quiet=True))
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.addMiner(Miner(100 * 2**20, is_quiet=True))
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.stopMiner(1)
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.stop()
    manager.run(inf)
    return manager

def _run1(daa, multiplier):
    manager = Manager(daa=daa)
    manager.start()
    manager.addMiner(Miner(25 * 2**20, is_quiet=True))
    manager.addMiner(Miner(multiplier * 25 * 2**20, is_quiet=True))
    manager.run(3600 * 24 * 2, show_progress=True)
    manager.stopMiner(1)
    manager.run(3600 * 24 * 400, until_ev_type=EventType.NEW_BLOCK, show_progress=True)
    manager.run(3600 * 24 * 4, show_progress=True)
    manager.stop()
    manager.run(inf)
    return manager

def run_1k(daa):
    return _run1(daa, 1000)

def run_1M(daa):
    return _run1(daa, 1000000)

daa_choices = {
    'lwma': lambda: LWMA(n=134, tl_rules=False),
    'htr': lambda: HTR(),
    'htr134': lambda: HTR(134),
    'crazy': lambda: CRAZY(n=134),
    'msb': lambda: MSB(n=134 * 2),
}

profile_choices = dict((f.__name__, f) for f in [
    run_1k,
    run_1M,
    run_4x,
])

parser = argparse.ArgumentParser()
parser.add_argument('daa', help='DAA algorithm', choices=list(daa_choices.keys()))
parser.add_argument('profile', help='Profile', choices=list(profile_choices.keys()))
parser.add_argument('n', type=int, help='number of loops')
parser.add_argument('--reset', action='store_true', help='overwrite previous results')
args = parser.parse_args()

daa = daa_choices[args.daa]()
profile = profile_choices[args.profile]

print('Running', args)

managers = []
for i in range(args.n):
    print('Loop {}'.format(i + 1))
    managers.append(profile(daa))

basename = './data/daa-{}-{}'.format(args.profile, args.daa)
filename = '{}.pickle'.format(basename)
if not args.reset and os.path.isfile(filename):
    previous = pickle.load(open(filename, 'rb'))
    managers = previous + managers

fp = open(filename, 'wb')
pickle.dump(managers, fp)
fp.close()
print('Saved to {} (total={} loops)'.format(filename, len(managers)))

img_filename = '{}.png'.format(basename)
utils.plot_difficulty(managers, save_to=img_filename)
os.system('open {}'.format(img_filename))

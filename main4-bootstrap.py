from simulator import Manager, Miner
from difficulty import HTR, LWMA, MSB, CRAZY
from math import inf, ceil
import os
import pickle
import argparse

#daa = LWMA(n=134, tl_rules=False)
#daa = HTR()
daa = MSB(n=134 * 2)
#daa = CRAZY(n=134)

def run():
    manager = Manager(daa=daa)
    manager.start()
    manager.addMiner(Miner(25 * 2**30, is_quiet=True))
    manager.addMiner(Miner(1000 * 25 * 2**30, is_quiet=True))
    manager.run(3600 * 24 * 1, show_progress=True)
    for _ in range(10):
        manager.miners[1].stop()
        manager.run(3600 * 8, show_progress=True)
        manager.miners[1].start()
        manager.run(3600 * 8, show_progress=True)
    manager.stop()
    manager.run(inf)
    return manager

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, help='number of loops')
parser.add_argument('--overwrite', action='store_true', help='overwrite previous results')

args = parser.parse_args()

managers = []
for i in range(args.n):
    print('Loop {}'.format(i + 1))
    managers.append(run())

filename = 'daa4-bootstrap-{}.pickle'.format(daa.__class__.__name__)
if not args.overwrite and os.path.isfile(filename):
    previous = pickle.load(open(filename, 'rb'))
    managers = previous + managers

fp = open(filename, 'wb')
pickle.dump(managers, fp)
fp.close()
print('Saved to {} (total={} loops)'.format(filename, len(managers)))

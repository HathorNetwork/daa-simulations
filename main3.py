from simulator import Manager, Miner
from difficulty import HTR, LWMA, MSB
from math import inf, ceil
import pickle

save_pickle = True

daa = LWMA(n=134, tl_rules=False)
#daa = HTR()
#daa = MSB(n=134 * 2)

manager = Manager(daa=daa)
manager.start()

manager.addMiner(Miner(25 * 2**20, is_quiet=True))
manager.run(3600 * 8, show_progress=True)

for _ in range(24):
    manager.addMiner(Miner(25 * 2**20, is_quiet=True))
    manager.run(3600, show_progress=True)

for i in range(24):
    manager.stopMiner(i + 1)
    manager.run(3600, show_progress=True)

manager.run(3600 * 8, show_progress=True)

manager.stop()
manager.run(inf)

if save_pickle:
    days = ceil(manager.seconds() / 3600 / 24)
    filename = 'daa3-{}-{}m-{}d.pickle'.format(daa.__class__.__name__, len(manager.miners), days)
    fp = open(filename, 'wb')
    pickle.dump(manager, fp)
    fp.close()
    print('Saved to {}'.format(filename))

from simulator import Manager, Miner
from difficulty import HTR, LWMA, MSB
from math import inf
import pickle

days = 3
save_pickle = True

#daa = LWMA(n=134, tl_rules=False)
daa = HTR()
#daa = MSB(n=134 * 2)

manager = Manager(daa=daa)
manager.start()

manager.addMiner(Miner(25 * 2**20, is_quiet=True))
manager.run(3600 * 24 * 5, show_progress=True)

manager.stop()
manager.run(inf)

if save_pickle:
    filename = 'daa0-{}-{}m-{}d.pickle'.format(daa.__class__.__name__, len(manager.miners), days)
    fp = open(filename, 'wb')
    pickle.dump(manager, fp)
    fp.close()
    print('Saved to {}'.format(filename))

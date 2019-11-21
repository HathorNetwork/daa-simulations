
import heapq
import random
from collections import defaultdict
from enum import Enum
from math import log
from typing import NamedTuple

import numpy.random

from utils import sum_weights


class EventType(Enum):
    NEW_BLOCK = 1
    BLOCK_PROPAGATION = 2


class DelayedCall:
    def __init__(self, ev_type, seconds, fn, args, kwargs):
        self.ev_type = ev_type
        self.seconds = seconds
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.active = True

    def __repr__(self):
        return 'DelayedCall({}, {}, {})'.format(self.ev_type, self.seconds, self.fn)

    def __gt__(self, other):
        return self.seconds > other.seconds

    def cancel(self):
        self.active = False

    def run(self):
        self.fn(*self.args, **self.kwargs)


class Neighbor(NamedTuple):
    miner: 'Miner'
    rtt: float
    sigma: float

    def get_random_delay(self) -> float:
        delay = random.normalvariate(self.rtt, self.sigma)
        if delay <= 0:
            return 0.001  # 1ms
        return delay


class Block(NamedTuple):
    hash: int
    timestamp: int
    weight: float
    height: int
    miner: 'Miner'
    parent: 'Block'
    logwork: float

    def get_blockchain(self):
        cur = self
        while cur:
            yield cur
            cur = cur.parent


class Miner:
    counter = 1
    name_prefix = 'Miner'

    def __init__(self, hashrate: int, *, is_quiet: bool = False) -> None:
        self.hashrate = hashrate
        self.manager = None
        self.known_blocks = {}   # Dict[int, Block]
        self.best_block = None
        self.neighbors = []
        self.name = '{} {}'.format(self.name_prefix, Miner.counter)
        self.block_timer = None
        self.is_running = False
        self.is_quiet = is_quiet
        Miner.counter += 1

    def start(self) -> None:
        self.is_running = True
        self.schedule_next_block()
        self.manager.onMinerStart(self)

    def stop(self) -> None:
        if self.block_timer:
            self.block_timer.cancel()
            self.block_timer = None
        self.is_running = False
        self.manager.onMinerStop(self)

    def get_blocks(self):
        return self.known_blocks.values()

    def schedule_next_block(self) -> None:
        weight, dt = self.manager.get_miner_next_block(self)

        seconds = self.manager.seconds()
        parent = self.best_block
        height = parent.height + 1
        logwork = sum_weights(parent.logwork, weight)
        timestamp = max(int(seconds + dt), parent.timestamp + 1)
        block = Block(
            hash=random.getrandbits(256),
            timestamp=timestamp,
            weight=weight,
            miner=self.name,
            height=height,
            parent=parent,logwork=logwork
        )

        self.block_timer = self.manager.callLater(EventType.NEW_BLOCK, dt, self.on_block_found, block)
    
    def on_new_best_block(self, new_best_block):
        self.best_block = new_best_block

    def on_block_found(self, block, propagated=False) -> None:
        if block.hash in self.known_blocks:
            return

        if self.block_timer:
            # print('Skipping...')
            self.block_timer.cancel()
            self.block_timer = None

        if not propagated and not self.is_quiet:
            dt = block.timestamp - block.parent.timestamp
            print('[{}] New block found: hash={} height={:4d} ts={:8.2f} dt={:6.2f} weight={:8.4f} logwork={:8.4f}'.format(self.name, block.hash, block.height, block.timestamp, dt, block.weight, block.logwork))

        self.known_blocks[block.hash] = block
        if block.logwork > self.best_block.logwork:
            self.on_new_best_block(block)

        if self.is_running:
            self.schedule_next_block()
        self.propagate_block(block)

    def get_neighbors(self) -> None:
        return self.neighbors

    def propagate_block(self, block) -> None:
        for neighbor in self.get_neighbors():
            dt = neighbor.get_random_delay()
            self.manager.callLater(EventType.BLOCK_PROPAGATION, dt, neighbor.miner.on_block_found, block, propagated=True)


class Miner51Attack(Miner):
    name_prefix = 'Miner51Attack'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_attacking = False
        self.buffer = []

    def start_attack(self, first_block=None):
        self.is_attacking = True
        if first_block:
            self.best_block = first_block

    def stop_attack(self):
        self.is_attacking = False
        self.best_block = max(self.known_blocks.values(), key=lambda x: x.logwork)
        self.flush_blocks()

    def on_new_best_block(self, new_best_block):
        if not self.is_attacking:
            self.best_block = new_best_block
        else:
            if new_best_block.miner == self.name:
                self.best_block = new_best_block

    def propagate_block(self, block) -> None:
        if self.is_attacking:
            self.buffer.append(block)
        else:
            super().propagate_block(block)

    def flush_blocks(self):
        for block in self.buffer:
            self.propagate_block(block)

class Manager:
    def __init__(self, daa, *, weight_decay=False):
        self.miners = []
        self.genesis = Block(
            hash=random.getrandbits(256),
            timestamp=0,
            weight=24,
            miner='Genesis',
            height=1,
            parent=None,
            logwork=24
        )
        self.total_hashrate = 0
        self.total_hashrate_history = defaultdict(int)   # Dict[int, int]  (timestamp, hashrate)
        self.events = []
        self._seconds = 0
        self.target = 30     # seconds
        self.daa = daa
        self.is_running = False
        self.weight_decay = weight_decay

    def getWeight(self, blocks) -> float:
        #return self.daa.next_weight((x.timestamp, x.weight) for x in blocks)
        return self.daa.next_weight(blocks)

    def seconds(self) -> int:
        return self._seconds

    def callLater(self, ev_type, delay, fn, *args, **kwargs) -> 'IDelayedCall':
        event = DelayedCall(ev_type, self.seconds() + delay, fn, args, kwargs)
        heapq.heappush(self.events, event)
        return event

    def addMiner(self, miner, *, is_synced: bool = True):
        miner.manager = self

        if is_synced and self.miners:
            miner.best_block = self.miners[0].best_block
            miner.known_blocks = self.miners[0].known_blocks.copy()
        else:
            miner.best_block = self.genesis

        for other in self.miners:
            other.neighbors.append(Neighbor(miner, 0.05, 0.01))
            miner.neighbors.append(Neighbor(other, 0.05, 0.01))
        self.miners.append(miner)
        if self.is_running:
            miner.start()

    def onMinerStart(self, miner):
        self.total_hashrate += miner.hashrate
        self.total_hashrate_history[self.seconds()] = self.total_hashrate

    def onMinerStop(self, miner):
        self.total_hashrate -= miner.hashrate
        self.total_hashrate_history[self.seconds()] = self.total_hashrate

    def stopMiner(self, index):
        miner = self.miners[index]
        miner.stop()

    def start(self) -> None:
        self.is_running = True
        for miner in self.miners:
            miner.start()

    def stop(self) -> None:
        self.is_running = False
        for miner in self.miners:
            miner.stop()

    def get_next_block_dt(self, weight, hashrate):
        if weight > 40:
            method = 'too_small'
            # p = 2**(-weight)
            # u = random.random()
            # Using the inverse transform sampling: attempts = log(u) / log(1 - p)
            # As p is really small, from the Taylor Series: log(1 - p) = -p
            # Thus, attempts = log(u) / (-2**(-weight)) = (2**weight) * (-log(u))
            attempts = (2**weight) * (-log(random.random()))
        else:
            method = 'geometric'
            geometric_p = 2**(-weight)
            attempts = numpy.random.geometric(geometric_p)
        dt = attempts / hashrate
        assert dt > 0, 'dt={} method={} attempts={} hashrate={}'.format(dt, method, attempts, hashrate)
        return dt

    def get_miner_next_block(self, miner):
        weight = self.getWeight(miner.best_block.get_blockchain())
        dt = self.get_next_block_dt(weight, miner.hashrate)
        if self.weight_decay:
            max_k = 300
            while dt > max_k * self.target:
                weight -= 2.73
                dt = max_k * self.target + self.get_next_block_dt(weight, miner.hashrate)
                max_k += 60
        return weight, dt

    def run(self, interval: float, *, until_ev_type = None, show_progress: bool = False) -> None:
        if show_progress:
            if interval > 3600 * 24:
                factor = 3600 * 24  # day
            elif interval > 3600:
                factor = 3600       # hour
            else:
                factor = 1
            from tqdm import tqdm
            pbar = tqdm(total=interval / factor)
        start = self.seconds()
        while self.seconds() - start < interval:
            if len(self.events) == 0:
                print('Finish!')
                break
            event = heapq.heappop(self.events)
            if event.active:
                if show_progress:
                    pbar.update((event.seconds - self._seconds) / factor)
                assert event.seconds > self._seconds, '{} should be higher than {} (ev={})'.format(self._seconds, event.seconds, event)
                self._seconds = event.seconds
                event.run()
                if event.ev_type == until_ev_type:
                    break

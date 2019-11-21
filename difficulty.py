from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, TypeVar


def weight_to_diff(w: float) -> int:
    return int(2**w)


def diff_to_weight(d: int) -> float:
    import math
    return math.log(d, 2) if d > 1 else 0


T = TypeVar('T')


def take(iterator: Iterator[T], n: int) -> List[T]:
    """Take first n elements from iterator, can return less if there aren't enough elements."""
    count = 0
    res = []
    for i in iterator:
        res.append(i)
        count += 1
        if count >= n:
            break
    return res


class DAA(ABC):
    @abstractmethod
    def next_weight(self, blocks: Iterator[Tuple[int, float]]) -> float:
        """Determines the next weight based on the list of block timestamps and their respective weights."""
        raise NotImplementedError


class HTR(DAA):
    T = 30
    N = 20
    MIN_WEIGHT = 21.0
    MAX_DW = 0.25

    def __init__(self, n=None, max_dw_rule=True, min_weight_rule=True, foo=False):
        self.n = self.N if n is None else n
        self.max_dw_rule = max_dw_rule
        self.min_weight_rule = min_weight_rule
        self.foo = foo

    def _get_n(self, blocks: Iterator['Block']) -> List['Block']:
        return take(blocks, self.n)[::-1]

    def apply_max_dw(self, new_weight: float, old_weight: float) -> float:
        if not self.max_dw_rule:
            return new_weight
        max_dw = self.MAX_DW
        dw = new_weight - old_weight
        if dw > max_dw:
            new_weight = old_weight + max_dw
        elif dw < -max_dw:
            new_weight = old_weight - max_dw
        return new_weight

    def apply_min_weight(self, weight: float) -> float:
        if not self.min_weight_rule:
            return weight
        if weight < self.MIN_WEIGHT:
            weight = self.MIN_WEIGHT
        return weight

    def next_weight(self, iter_blocks: Iterator['Block']) -> float:
        import math
        from utils import sum_weights

        blocks = self._get_n(iter_blocks)

        if len(blocks) < 2:
            return self.MIN_WEIGHT

        dt = blocks[-1].timestamp - blocks[0].timestamp
        if dt < 1:
            dt = 1
        #assert dt > 0

        logH = 0.0
        for blk in blocks:
            logH = sum_weights(logH, blk.weight)
        weight = logH - math.log(dt, 2) + math.log(self.T, 2)

        # apply a maximum change in weight
        weight = self.apply_max_dw(weight, blocks[-1].weight)

        # apply min weight
        weight = self.apply_min_weight(weight)

        return weight


class LWMA(DAA):
    T = 30     # masari/cryptonote: DIFFICULTY_TARGET = 60 // seconds
    N = 134    # masari/cryptonote: DIFFICULTY_WINDOW = 720 // blocks
    FTL = 300  # masari/cryptonote: BLOCK_FUTURE_TIME_LIMIT = DIFFICULTY_TARGET * 5
    PTL = 300  # masari/cryptonote: BLOCK_PAST_TIME_LIMIT = DIFFICULTY_TARGET * 5
    MIN_WEIGHT = 21.0
    MIN_DIFF = weight_to_diff(MIN_WEIGHT)
    MIN_LWMA = T // 4  # =7

    # To get an average solvetime to within +/- ~0.1%, use an adjustment factor.
    # adjust=0.998 for N = 60  TODO: recalculate for N = 30
    _ADJUST = 0.998

    def __init__(self, n=None, harmonic=True, tl_rules=True, *, debug=False):
        self.debug = debug
        self.harmonic = harmonic
        self.tl_rules = tl_rules
        self.n = self.N if n is None else n

    def _get_solvetimes_and_diffs(self, blocks: Iterator['Block']) -> List[Tuple[int, int]]:
        from itertools import accumulate
        blocks = take(blocks, self.n + 1)[::-1]
        solvetimes = [b1.timestamp - b0.timestamp for b1, b0 in zip(blocks[1:], blocks[:-1])]
        diffs = list(map(weight_to_diff, (blk.weight for blk in blocks)))
        return list(zip(solvetimes, diffs[1:]))

    def next_weight(self, blocks: Iterator[Tuple[int, float]]) -> float:
        solvetimes_and_diffs = self._get_solvetimes_and_diffs(blocks)

        # Return a difficulty of 1 for first 3 blocks if it's the start of the chain.
        if len(solvetimes_and_diffs) < 3:
            return self.MIN_WEIGHT

        solvetimes, difficulties = zip(*solvetimes_and_diffs)
        N = self.n

        # Otherwise, use a smaller N if the start of the chain is less than N+1.
        if len(solvetimes) < N:
            N = len(solvetimes) - 1
        # Otherwise make sure solvetimes and difficulties are correct size.
        else:
            assert len(solvetimes) == len(difficulties) == N

        # double LWMA(0), sum_inverse_D(0), harmonic_mean_D(0), nextDifficulty(0);
        # uint64_t difficulty(0), next_difficulty(0);
        LWMA = 0.0
        sum_inverse_diff = 0.0
        sum_diff = 0.0

        # The divisor k normalizes the LWMA sum to a standard LWMA.
        k = N * (N + 1) / 2

        # Loop through N most recent blocks. N is most recently solved block.
        for i in range(N):
            solvetime = solvetimes[i]
            if self.tl_rules:
                solvetime = min(self.PTL, max(solvetime, -self.FTL))
            difficulty = difficulties[i]
            LWMA += solvetime * (i + 1) / k
            sum_inverse_diff += 1 / difficulty
            sum_diff += difficulty

        harmonic_mean_diff = N / sum_inverse_diff
        arithmetic_mean_diff = sum_diff  / N
        if self.harmonic:
            mean_diff = harmonic_mean_diff
        else:
            mean_diff = arithmetic_mean_diff

        # Limit LWMA same as Bitcoin's 1/4 in case something unforeseen occurs.
        if int(LWMA) < self.MIN_LWMA:
            LWMA = float(self.MIN_LWMA)

        next_diff = mean_diff * self.T / LWMA * self._ADJUST
        # No limits should be employed, but this is correct way to employ a 20% symmetrical limit:
        # next_diff = max(prev_diff * 0.8, min(prev_diff / 0.8, next_diff))
        next_difficulty = int(next_diff)

        #if next_difficulty == 0:
        #    return self.MIN_WEIGHT
        next_weight = diff_to_weight(next_difficulty)
        if self.debug:
            print(next_weight, next_difficulty, int(harmonic_mean_diff), LWMA)

        next_weight = max(next_weight, self.MIN_WEIGHT)
        return next_weight


class SE(HTR):
    def next_weight(self, blocks: Iterator[Tuple[int, float]]) -> float:
        import math
        timestamps, weights = zip(*take(blocks, 2))

        if len(timestamps) < 2:
            return self.MIN_WEIGHT

        solvetime = timestamps[-1] - timestamps[-2]
        weight = weights[-1] + (self.T - solvetime) / self.n / math.log(2)

        # apply a maximum change in weight
        weight = self.apply_max_dw(weight, weights[-1])

        # apply min weight
        weight = self.apply_min_weight(weight)

        return weight


class CRAZY(HTR):
    def next_weight(self, blocks: Iterator['Block']) -> float:
        import math

        block = next(blocks)
        height, time_since_genesis = block.height, block.timestamp

        weight = (height - time_since_genesis / self.T) / self.n / math.log(2)

        # apply a maximum change in weight
        try:
            prev_block = next(blocks)
            weight = self.apply_max_dw(weight, prev_block.weight)
        except StopIteration:
            pass

        # apply min weight
        weight = self.apply_min_weight(weight)

        return weight


class MSB(LWMA):
    S = 5

    def _get_solvetimes_and_weights(self, blocks: Iterator['Block']) -> List[Tuple[int, int]]:
        from itertools import accumulate
        blocks = take(blocks, self.n + 1)[::-1]
        solvetimes = [b1.timestamp - b0.timestamp for b1, b0 in zip(blocks[1:], blocks[:-1])]
        diffs = list(blk.weight for blk in blocks)
        return list(zip(solvetimes, diffs[1:]))

    def next_weight(self, blocks: Iterator[Tuple[int, float]]) -> float:
        import math
        from utils import sum_weights

        solvetimes_and_weights = self._get_solvetimes_and_weights(blocks)

        # Return a difficulty of 1 for first 3 blocks if it's the start of the chain.
        if len(solvetimes_and_weights) < 10:
            return self.MIN_WEIGHT

        solvetimes, weights = zip(*solvetimes_and_weights)
        N = self.n

        # Otherwise, use a smaller N if the start of the chain is less than N+1.
        if len(solvetimes) < N:
            N = len(solvetimes) - 1
        # Otherwise make sure solvetimes and difficulties are correct size.
        else:
            assert len(solvetimes) == len(weights) == N
        #total_solvetimes = sum(solvetimes)

        K = N // 2

        sum_diffs = 0.0
        sum_solvetimes = 0.0

        prefix_sum_solvetimes = [0]
        for x in solvetimes:
            prefix_sum_solvetimes.append(prefix_sum_solvetimes[-1] + x)

        # Loop through N most recent blocks. N is most recently solved block.
        for i in range(K, N):
            solvetime = solvetimes[i]
            weight = weights[i]

            #x = sum(solvetimes[i - K:i + 1]) / K
            #assert sum(solvetimes[i - K:i + 1]) == prefix_sum_solvetimes[i + 1] - prefix_sum_solvetimes[i - K]
            x = (prefix_sum_solvetimes[i + 1] - prefix_sum_solvetimes[i - K]) / K

            ki = K * (x - self.T)**2 / (2 * self.T * self.T)
            ki = max(1, ki / self.S)
            if self.debug and ki > 1:
                print('outlier!!!', i, ki, x) #solvetime, weight, i)
            #ki = i - K + 2
            sum_diffs += ki * weight_to_diff(weight)
            sum_solvetimes += ki * solvetime

        weight = math.log(sum_diffs, 2) - math.log(sum_solvetimes, 2) + math.log(self.T, 2)

        # apply a maximum change in weight
        #weight = self.apply_max_dw(weight, prev_weight)

        # apply min weight
        #weight = self.apply_min_weight(weight)

        return weight

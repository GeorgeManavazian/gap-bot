import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from sector_cap import make_resting_orders, sector_counts
from live.engine import resting_orders  # pure function

# (1) property: huge cap == engine.resting_orders
rng = random.Random(0)
smap_all = {f"T{i}": f"S{i % 4}" for i in range(30)}
f = make_resting_orders(10**9, smap_all)
for _ in range(200):
    n = rng.randint(0, 25)
    live = rng.sample(sorted(smap_all), n)
    # coarse gaps force ties -> exercises ticker tiebreak
    pending = {t: {"gap_pct": -float(rng.randint(2, 6))} for t in live}
    free = rng.randint(-1, 12)
    opn = set(rng.sample(sorted(smap_all), rng.randint(0, 5)))
    assert f(live, pending, free, opn) == resting_orders(live, pending, free), (live, free)

# (2) cap=1 skipping
smap = {"A": "X", "B": "X", "C": "Y", "D": "X", "E": "Y", "F": "Z"}
pending = {"A": {"gap_pct": -9}, "B": {"gap_pct": -8}, "C": {"gap_pct": -7},
           "D": {"gap_pct": -6}, "E": {"gap_pct": -5}, "F": {"gap_pct": -4}}
live = list(pending)
g = make_resting_orders(1, smap)
assert g(live, pending, 3, set()) == {"A", "C", "F"}       # B,D skipped (X full), E skipped (Y full)
assert g(live, pending, 5, set()) == {"A", "C", "F"}       # list ends early
assert g(live, pending, 3, {"A"}) == {"C", "F"}            # held A fills sector X
assert g(live, pending, 2, {"C"}) == {"A", "F"}            # held C fills Y; E skipped
assert g(live, pending, 0, set()) == set()
assert make_resting_orders(2, smap)(live, pending, 4, set()) == {"A", "B", "C", "E"}

# (3) unmapped tickers uncapped
u = make_resting_orders(1, {})
assert u(live, pending, 4, {"A", "B"}) == {"A", "B", "C", "D"}
assert sector_counts(["A", "B", "C", "Q"], smap) == {"X": 2, "Y": 1, "Unknown": 1}
print("test_sector_cap: all passed")

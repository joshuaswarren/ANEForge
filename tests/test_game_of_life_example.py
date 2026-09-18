"""The Game of Life example's on-engine rule against a numpy reference. Requires the ANE."""
import sys
from pathlib import Path

import numpy as np

from _helpers import requires_ane

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import game_of_life as gol  # noqa: E402

pytestmark = requires_ane  # the step program dispatches to the engine


def test_life_step_matches_numpy():
  n = 16
  prog = gol.step_program(n)
  onames = [name for _, name in prog.output_ports]
  rng = np.random.default_rng(0)
  alive = (rng.random((n, n)) < 0.3).astype(np.float32).reshape(1, 1, n, n)
  ref = alive.copy()
  try:
    for _ in range(5):
      alive = prog(alive)[onames[0]].reshape(1, 1, n, n)
      ref = gol.numpy_step(ref)
      assert np.array_equal((alive > 0.5).ravel(), (ref > 0.5).ravel())
  finally:
    prog.release()


def test_life_blinker_oscillates():
  n = 8
  alive = np.zeros((1, 1, n, n), np.float32)
  alive[0, 0, 3, 2:5] = 1.0                                  # a horizontal blinker, away from the zero-padded border
  prog = gol.step_program(n)
  onames = [name for _, name in prog.output_ports]
  try:
    one = prog(alive)[onames[0]].reshape(1, 1, n, n)
    two = prog(one)[onames[0]].reshape(1, 1, n, n)
    assert not np.array_equal(one, alive)                    # period 2: it flips to vertical ...
    assert np.array_equal(two, alive)                        # ... and back
  finally:
    prog.release()

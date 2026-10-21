"""`Tensor.silu_tanh` is `x * sigmoid(x)` as `h * (1 + tanh(h))`, h = x / 2. No model site uses it now (the Qwen3.8 decode uses `silu_expdiv`, see
tests/test_silu_expdiv.py). Off-device: the graph, the emitted MIL and a numpy evaluation."""
import re

import numpy as np

from aneforge import _compile
from aneforge.graph import input as _input

def _topo(roots):
  seen, order = set(), []

  def visit(t):
    if id(t) in seen: return
    seen.add(id(t))
    for s in t.srcs: visit(s)
    order.append(t)
  for r in roots: visit(r)
  return order


def _ops(roots):
  return [t.op for t in _topo(roots)]


def _evaluate(root, feed):
  val = {}
  for t in _topo([root]):
    a = [val[id(s)] for s in t.srcs]
    if t.op == "input": v = feed[id(t)]
    elif t.op == "muls": v = a[0] * t.attrs["k"]
    elif t.op == "adds": v = a[0] + t.attrs["k"]
    elif t.op == "tanh": v = np.tanh(a[0])
    elif t.op == "mul": v = a[0] * a[1]
    else: raise AssertionError(f"evaluator has no rule for {t.op}")
    val[id(t)] = v
  return val[id(root)]


def test_graph_is_scalar_multiply_tanh_scalar_add_multiply():
  x = _input((1, 64))
  y = x.silu_tanh()
  ops = [t.op for t in _topo([y]) if t.op != "input"]
  assert ops == ["muls", "tanh", "adds", "mul"], ops
  assert y.shape == (1, 64)
  by_op = {t.op: t for t in _topo([y])}
  assert by_op["muls"].attrs["k"] == 0.5 and by_op["adds"].attrs["k"] == 1.0


def test_emitted_mil_has_tanh_and_no_native_silu():
  y = _input((1, 64)).silu_tanh()
  mil = (_compile._lower_fused_to_dir(y) / "model.mil").read_text()
  assert mil.count("tanh(") == 1 and "silu(" not in mil
  assert len(re.findall(r"val = fp16\(0x1\.0+p-1\)", mil)) == 1 and len(re.findall(r"val = fp16\(0x1\.0+p\+0\)", mil)) == 1


def test_numpy_evaluation_equals_x_times_sigmoid_x():
  xs = np.concatenate([np.linspace(-16, 16, 6001), [0.0, 1e-4, -1e-4, 50.0, -50.0, 700.0, -700.0]])
  x = _input((1, len(xs)))
  got = _evaluate(x.silu_tanh(), {id(x): xs})
  with np.errstate(over="ignore"):
    want = xs / (1.0 + np.exp(-xs))
  assert np.allclose(got, want, rtol=1e-12, atol=1e-12), np.abs(got - want).max()

"""The state block of the Gated-DeltaNet decode runs its three matmuls (k @ S, k^T @ delta, q @ S) with the left operand
scaled by 2^8 and the result by 2^-8. Off-device: checks the graph shape, the emitted MIL constants, and evaluates the
graph with numpy to show the matmul sees the scaled operand while the block output is unchanged."""
import re
from types import SimpleNamespace
import numpy as np
from aneforge import _compile
from aneforge.graph import input as _input
from aneforge.qwen35 import _deltanet_decode_stage_b

NK, NV, DK, DV = 2, 4, 8, 8
CFG = SimpleNamespace(extra={"nk": NK, "nv": NV, "dk": DK, "dv": DV, "conv_k": 4})


def _build():
  lanes = {"x": _input((1, 16)), "q": _input((NV, 1, DK)), "k": _input((NV, 1, DK)), "v": _input((NV, 1, DV)),
           "beta": _input((NV, 1, 1)), "gt": _input((NV, 1, 1)), "z": _input((NV, DV))}
  out, pairs = _deltanet_decode_stage_b(lanes, {}, CFG, None, {}, 1)
  return lanes, out["o"], pairs[0][0], pairs[0][1]       # lanes, readout o, Sout, S


def _topo(t, seen=None, order=None):
  seen = set() if seen is None else seen
  order = [] if order is None else order
  if id(t) in seen: return order
  seen.add(id(t))
  for s in t.srcs: _topo(s, seen, order)
  order.append(t)
  return order


def _evaluate(roots, feed):
  """numpy float64 evaluation of the op subset the state block uses; returns {id(node): array}."""
  val = {}
  for t in _topo(roots[0]) + [n for r in roots[1:] for n in _topo(r)]:
    if id(t) in val: continue
    a = [val[id(s)] for s in t.srcs]
    if t.op == "input": v = feed[id(t)]
    elif t.op == "const_array": v = np.asarray(t.attrs["value"], np.float64)
    elif t.op == "muls": v = a[0] * t.attrs["k"]
    elif t.op == "mul": v = a[0] * a[1]
    elif t.op == "sub": v = a[0] - a[1]
    elif t.op == "add": v = a[0] + a[1]
    elif t.op == "reshape": v = a[0].reshape(t.shape)
    elif t.op == "transpose": v = a[0].transpose(t.attrs["perm"])
    elif t.op == "bmm": v = a[0] @ a[1]
    else: raise AssertionError(f"evaluator has no rule for {t.op}")
    val[id(t)] = v
  return val


def test_each_matmul_is_scaled_up_before_and_down_after():
  _, o, sout, _ = _build()
  nodes = list({id(t): t for t in _topo(o) + _topo(sout)}.values())
  bmms = {id(t): t for t in nodes if t.op == "bmm"}
  assert len(bmms) == 3
  users = {}
  for t in nodes:
    for s in t.srcs: users.setdefault(id(s), []).append(t)
  for t in bmms.values():
    left = t.srcs[0]
    assert left.op == "muls" and left.attrs["k"] == 256.0, "the left operand of a state-block matmul is not scaled by 2^8"
    down = users[id(t)]
    assert len(down) == 1 and down[0].op == "muls" and down[0].attrs["k"] == 1.0 / 256.0, "the matmul result is not scaled by 2^-8"


def test_emitted_mil_carries_both_constants_around_each_matmul():
  _, o, _, _ = _build()                                  # o depends on Sout, so all three matmuls are in the program
  mil = (_compile._lower_fused_to_dir(o) / "model.mil").read_text()
  const = {m.group(1): float.fromhex(m.group(2)) for m in re.finditer(r'fp16 (\w+) = const\(\)\[name = string\("\w+"\), val = fp16\((0x[^)]+)\)\]', mil)}
  mul = {m.group(1): m.group(3) for m in re.finditer(r'(\w+) = mul\(x = (\w+), y = (\w+)\)', mil)}      # result -> scale constant name
  mul_in = {m.group(1): m.group(2) for m in re.finditer(r'(\w+) = mul\(x = (\w+), y = (\w+)\)', mil)}   # result -> its input
  matmuls = re.findall(r'(\w+) = matmul\(transpose_x = bool\(false\), transpose_y = bool\(false\), x = (\w+), y = (\w+)\)', mil)
  assert len(matmuls) == 3
  for out, x, _ in matmuls:
    assert const[mul[x]] == 256.0, "the left matmul operand is not multiplied by the 2^8 constant"
    down = [r for r, src in mul_in.items() if src == out]
    assert len(down) == 1 and const[mul[down[0]]] == 1.0 / 256.0, "the matmul result is not multiplied by the 2^-8 constant"


def test_graph_value_equals_the_unscaled_recurrence_and_matmul_sees_the_scaled_operand():
  lanes, o, sout, s_in = _build()
  rng = np.random.default_rng(0)
  feed = {id(lanes[n]): rng.normal(0, 0.3, lanes[n].shape) for n in lanes}
  feed[id(s_in)] = rng.normal(0, 0.05, s_in.shape)
  v = _evaluate([o, sout], feed)
  q, k, vv = (feed[id(lanes[n])] for n in ("q", "k", "v"))
  beta, gt, S = feed[id(lanes["beta"])], feed[id(lanes["gt"])], feed[id(s_in)]
  S1 = S * gt
  delta = (vv - k @ S1) * beta
  want_S = S1 + k.transpose(0, 2, 1) @ delta
  want_o = (q @ want_S).reshape(NV, DV)
  assert np.allclose(v[id(sout)], want_S, rtol=1e-12, atol=1e-12)
  assert np.allclose(v[id(o)], want_o, rtol=1e-12, atol=1e-12)
  first = [t for t in _topo(sout) if t.op == "bmm"][0]            # k @ S1: the raw matmul output is 256x the product
  assert np.allclose(v[id(first)], 256.0 * (k @ S1), rtol=1e-12, atol=1e-12)

"""`real_div` has a VJP, so a graph through `Tensor.silu_expdiv` (the SwiGLU gate of `llm._swiglu`, the Gated-DeltaNet conv silu) can be differentiated.
Off-device: the backward graph is built and evaluated in numpy against central differences, and a dense SwiGLU block is differentiated end to end."""
from types import SimpleNamespace

import numpy as np

from aneforge import autograd as agrad
from aneforge.llm import _swiglu


def _topo(root):
  seen, order = set(), []
  stack = [(root, False)]
  while stack:
    t, done = stack.pop()
    if done:
      order.append(t)
      continue
    if id(t) in seen: continue
    seen.add(id(t))
    stack.append((t, True))
    for s in t.srcs: stack.append((s, False))
  return order


def _evaluate(root, dtype=np.float64):
  val = {}
  for t in _topo(root):
    a = [val[id(s)] for s in t.srcs]
    op = t.op
    if op == "input": v = t.attrs["value"].astype(dtype)
    elif op == "muls": v = a[0] * t.attrs["k"]
    elif op == "adds": v = a[0] + t.attrs["k"]
    elif op == "add": v = a[0] + a[1]
    elif op == "sub": v = a[0] - a[1]
    elif op == "mul": v = a[0] * a[1]
    elif op == "real_div": v = a[0] / a[1]
    elif op == "exp": v = np.exp(a[0])
    elif op == "sigmoid": v = 1 / (1 + np.exp(-a[0]))
    elif op == "reduce_sum": v = a[0].sum(tuple(t.attrs["axes"]), keepdims=True) if "axes" in t.attrs else a[0].sum(keepdims=True)
    elif op == "reshape": v = a[0].reshape(t.shape)
    else: raise AssertionError(f"evaluator has no rule for {op}")
    val[id(t)] = v
  return val[id(root)]


def test_real_div_gradient_matches_finite_differences():
  rng = np.random.default_rng(1)
  av = rng.uniform(-2, 2, (3, 5)).astype(np.float32)
  bv = rng.uniform(0.5, 2, (3, 5)).astype(np.float32)
  wv = rng.normal(0, 1, (3, 5)).astype(np.float32)
  a, b = agrad.parameter(av), agrad.parameter(bv)
  loss = ((a / b) * agrad.parameter(wv)).sum((0, 1))
  g = agrad.backward(loss, [a, b])
  fd_a = wv / bv
  fd_b = -wv * av / bv ** 2
  assert np.allclose(_evaluate(g[a]), fd_a, rtol=1e-5, atol=1e-6)
  assert np.allclose(_evaluate(g[b]), fd_b, rtol=1e-5, atol=1e-6)


def test_real_div_gradient_sums_over_a_broadcast_divisor():
  rng = np.random.default_rng(4)
  av = rng.uniform(-2, 2, (3, 5)).astype(np.float32)
  bv = rng.uniform(0.5, 2, (1, 5)).astype(np.float32)
  wv = rng.normal(0, 1, (3, 5)).astype(np.float32)
  a, b = agrad.parameter(av), agrad.parameter(bv)
  g = agrad.backward(((a / b) * agrad.parameter(wv)).sum((0, 1)), [a, b])
  assert g[b].shape == (1, 5) and g[a].shape == (3, 5)
  assert np.allclose(_evaluate(g[a]), wv / bv, rtol=1e-5, atol=1e-6)
  assert np.allclose(_evaluate(g[b]), (-wv * av / bv ** 2).sum(0, keepdims=True), rtol=1e-5, atol=1e-6)


def test_real_div_gradient_sums_over_a_broadcast_numerator():
  rng = np.random.default_rng(5)
  av = rng.uniform(-2, 2, (1, 4)).astype(np.float32)
  bv = rng.uniform(0.5, 2, (3, 4)).astype(np.float32)
  wv = rng.normal(0, 1, (3, 4)).astype(np.float32)
  a, b = agrad.parameter(av), agrad.parameter(bv)
  g = agrad.backward(((a / b) * agrad.parameter(wv)).sum((0, 1)), [a, b])
  assert g[a].shape == (1, 4) and g[b].shape == (3, 4)
  assert np.allclose(_evaluate(g[a]), (wv / bv).sum(0, keepdims=True), rtol=1e-5, atol=1e-6)
  assert np.allclose(_evaluate(g[b]), -wv * av / bv ** 2, rtol=1e-5, atol=1e-6)


def test_silu_expdiv_gradient_is_silu_prime():
  xv = np.linspace(-20, 20, 40, dtype=np.float32).reshape(4, 10)
  wv = np.random.default_rng(2).normal(0, 1, xv.shape).astype(np.float32)
  x = agrad.parameter(xv)
  g = agrad.backward((x.silu_expdiv() * agrad.parameter(wv)).sum((0, 1)), [x])
  s = 1 / (1 + np.exp(-xv.astype(np.float64)))
  want = wv * s * (1 + xv * (1 - s))
  assert np.allclose(_evaluate(g[x]), want, rtol=1e-5, atol=1e-6), np.abs(_evaluate(g[x]) - want).max()


def test_silu_expdiv_gradient_is_finite_in_fp16_below_the_exp_overflow():
  """exp(-x) is infinite in fp16 for x < -11.09. The gradient graph must not pass through it (0 * inf is NaN)."""
  xv = np.array([[-30.0, -20.0, -12.0, -11.0, -3.0, 0.0, 3.0, 12.0, 20.0, 30.0]], np.float32)
  x = agrad.parameter(xv)
  g = agrad.backward(x.silu_expdiv().sum((0, 1)), [x])
  with np.errstate(all="ignore"):
    got = _evaluate(g[x], np.float16)
  assert np.isfinite(got).all(), got
  s = 1 / (1 + np.exp(-xv.astype(np.float64)))
  assert np.allclose(got, s * (1 + xv * (1 - s)), atol=2e-3), got


def test_dense_swiglu_block_can_be_differentiated():
  rng = np.random.default_rng(3)
  d, f = 8, 16
  cfg = SimpleNamespace(norm_eps=1e-6)
  w = {"mlp_norm": np.ones(d, np.float32), "wgate": rng.normal(0, 0.1, (f, d)).astype(np.float32),
       "wup": rng.normal(0, 0.1, (f, d)).astype(np.float32), "wdown": rng.normal(0, 0.1, (d, f)).astype(np.float32)}
  h = agrad.parameter(rng.normal(0, 1, (1, d)))
  g = agrad.backward(_swiglu(h, w, cfg, None).sum((0, 1)), [h])
  assert set(g) == {h}

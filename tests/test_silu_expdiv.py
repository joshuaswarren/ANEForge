"""`Tensor.silu_expdiv` is `x * sigmoid(x)` as `x / (exp(-x) + 1)`. The Qwen3.8 decode uses it at the Gated-DeltaNet conv silu and at the SwiGLU gate of every layer
(the decode and the speculative verifier) and keeps the native silu at the z gate. Off-device: the graph, the emitted MIL, a numpy evaluation, and the silu op
at each site."""
import re
from types import SimpleNamespace

import numpy as np

from aneforge import _compile
from aneforge.graph import input as _input
from aneforge.llm import _swiglu
from aneforge.qwen35 import _deltanet_decode_stage_a, _deltanet_decode_stage_c

D, NK, NV, DK, DV, K = 16, 2, 4, 8, 8, 4
CD = NK * DK * 2 + NV * DV
RNG = np.random.default_rng(0)
CFG = SimpleNamespace(norm_eps=1e-6, extra={"nk": NK, "nv": NV, "dk": DK, "dv": DV, "conv_k": K})


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
    elif t.op == "exp": v = np.exp(a[0])
    elif t.op == "real_div": v = a[0] / a[1]
    else: raise AssertionError(f"evaluator has no rule for {t.op}")
    val[id(t)] = v
  return val[id(root)]


def test_graph_is_scalar_multiply_exp_scalar_add_divide():
  x = _input((1, 64))
  y = x.silu_expdiv()
  ops = [t.op for t in _topo([y]) if t.op != "input"]
  assert ops == ["muls", "exp", "adds", "real_div"], ops
  assert y.shape == (1, 64)
  by_op = {t.op: t for t in _topo([y])}
  assert by_op["muls"].attrs["k"] == -1.0 and by_op["adds"].attrs["k"] == 1.0
  assert by_op["real_div"].srcs[0] is x            # the numerator is the input itself


def test_emitted_mil_has_exp_and_divide_and_no_native_silu():
  y = _input((1, 64)).silu_expdiv()
  mil = (_compile._lower_fused_to_dir(y) / "model.mil").read_text()
  assert mil.count("exp(") == 1 and mil.count("real_div(") == 1 and "silu(" not in mil
  assert len(re.findall(r"val = fp16\(-0x1\.0+p\+0\)", mil)) == 1 and len(re.findall(r"val = fp16\(0x1\.0+p\+0\)", mil)) == 1


def test_numpy_evaluation_equals_x_times_sigmoid_x():
  xs = np.concatenate([np.linspace(-16, 16, 6001), [0.0, 1e-4, -1e-4, 50.0, -50.0, 700.0]])
  x = _input((1, len(xs)))
  got = _evaluate(x.silu_expdiv(), {id(x): xs})
  want = xs / (1.0 + np.exp(-xs))
  assert np.allclose(got, want, rtol=1e-12, atol=1e-12), np.abs(got - want).max()


def _w_swiglu(f=32):
  return {"mlp_norm": np.ones(D, np.float32), "wgate": RNG.normal(0, 0.1, (f, D)).astype(np.float32), "wup": RNG.normal(0, 0.1, (f, D)).astype(np.float32),
          "wdown": RNG.normal(0, 0.1, (D, f)).astype(np.float32)}


def test_swiglu_gate_uses_the_expdiv_form():
  ops = _ops([_swiglu(_input((1, D)), _w_swiglu(), CFG, None)])
  assert "silu" not in ops and "tanh" not in ops and ops.count("exp") == 1 and ops.count("real_div") == 1


def _w_deltanet():
  return {"in_norm": np.ones(D, np.float32), "in_proj_qkv": RNG.normal(0, 0.1, (CD, D)).astype(np.float32), "in_proj_z": RNG.normal(0, 0.1, (NV * DV, D)).astype(np.float32),
          "in_proj_b": RNG.normal(0, 0.1, (NV, D)).astype(np.float32), "in_proj_a": RNG.normal(0, 0.1, (NV, D)).astype(np.float32),
          "conv1d": RNG.normal(0, 0.1, (CD, K)).astype(np.float32), "dt_bias": np.zeros(NV, np.float32), "neg_exp_A": -np.ones(NV, np.float32),
          "ssm_norm": np.ones(DV, np.float32), "out_proj": RNG.normal(0, 0.1, (D, NV * DV)).astype(np.float32)}


def test_deltanet_conv_silu_is_expdiv_and_the_z_gate_is_native():
  w = _w_deltanet()
  lanes, pairs = _deltanet_decode_stage_a({"x": _input((1, D))}, w, CFG, None, {}, 1)
  ops_a = _ops([lanes[k] for k in ("q", "k", "v", "beta", "gt", "z")] + [p[0] for p in pairs])
  assert "silu" not in ops_a and "tanh" not in ops_a
  assert ops_a.count("real_div") == 1                     # the one expdiv silu: the conv
  hmap = {"x": _input((1, D)), "o": _input((NV, DV)), "z": _input((NV, DV))}
  out, _ = _deltanet_decode_stage_c(hmap, w, CFG, None, {}, 1)
  ops_c = _ops([out["h"]])
  assert ops_c.count("silu") == 1 and "tanh" not in ops_c and "exp" not in ops_c


def test_speculative_verifier_uses_the_same_activation_as_the_decode():
  import inspect
  from aneforge import speculative
  src = inspect.getsource(speculative._verify_block)
  assert ".silu_expdiv()" in src and ".silu()" not in src and ".silu_tanh()" not in src

"""aneforge showpiece: Conway's Game of Life with every generation as ONE on-engine forward pass - the 8-neighbour count is a fixed 3x3 conv and the birth/survival rule is elementwise. Writes docs/assets/game_of_life.webp if Pillow is present. Run: python3 examples/game_of_life.py"""
import sys
import time
from pathlib import Path

import _common   # noqa: F401  (sets env + repo-root path; import before aneforge)
import numpy as np
import aneforge as af
from aneforge import _compile as _c

# brand palette for the console
TEAL, RUST, DIM, BOLD, GREY, R = (
    "\033[38;2;72;187;170m", "\033[38;2;235;130;70m", "\033[2m",
    "\033[1m", "\033[38;2;150;150;150m", "\033[0m")
CHECK = f"{TEAL}OK{R}"

# simulation parameters
N = 192                 # grid; the border is zero-padded, so the field is finite
STEPS = 100             # generations
FRAME_STRIDE = 2        # capture an animation frame every this many generations
DENSITY = 0.32          # random initial fill
ANE_RAIL_W = 1.48       # measured sustained ANE rail, W (see demos/power_efficiency.py)
ANIM_SIZE, ANIM_MS = 192, 70
ANIM_HOLD = 900


def out(s=""):
    sys.stdout.write(s + "\n"); sys.stdout.flush()


def secs(s):
    """One generation is ~0.1 ms, so a whole 100-step run rounds to '0.0s' in seconds."""
    return f"{s * 1e3:.0f} ms" if s < 1.0 else f"{s:.1f}s"


def joules(j):
    return f"{j * 1e3:.0f} mJ" if j < 1.0 else f"{j:.1f} J"


def neighbor_kernel():
    """The 8-neighbour count as a fixed 3x3 conv: all ones except the centre."""
    k = np.ones((1, 1, 3, 3), np.float32)
    k[0, 0, 1, 1] = 0.0
    return k


def step_program(n=N):
    """Build ONE e5rt program: alive -> alive' for a single Life generation.

    `(nb - k).abs().clip(0, 1)` is 0 when the count equals k and 1 otherwise, so
    `1 - that` is the exact-match indicator; the rule then needs no equality op.
    """
    alive = af.input((1, 1, n, n))
    nb = af.conv(alive, neighbor_kernel(), pad=1)                # live neighbours, 0..8 (border zero-padded)
    is2 = ((nb - 2.0).abs().clip(0.0, 1.0) * -1.0).adds(1.0)     # 1 where nb == 2
    is3 = ((nb - 3.0).abs().clip(0.0, 1.0) * -1.0).adds(1.0)     # 1 where nb == 3
    survive = alive * is2                                       # a live cell with exactly 2 neighbours survives
    born = is3                                                  # a dead cell with exactly 3 neighbours is born
    return _c.compile_multi([af.maximum(survive, born)])


def numpy_step(alive):
    """Reference Life step with the same zero-padding, for the spot check."""
    a = np.asarray(alive, np.float32)
    f = a.reshape(a.shape[-2:])                                  # the 2-D field (input may be [1,1,H,W])
    p = np.pad(f, 1)
    nb = np.zeros_like(f)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                nb += p[1 + di:1 + di + f.shape[0], 1 + dj:1 + dj + f.shape[1]]
    out = ((nb == 3) | ((f > 0.5) & (nb == 2))).astype(np.float32)
    return out.reshape(a.shape)


def seed():
    rng = np.random.default_rng(7)
    return (rng.random((N, N)) < DENSITY).astype(np.float32).reshape(1, 1, N, N)


def main():
    out()
    out(f"  {BOLD}{TEAL}ANEForge{R}  {DIM} - Conway's Game of Life on the Apple Neural Engine{R}")
    out(f"  {DIM}a 3x3 neighbour conv + an elementwise birth/survival rule, one fused program per step{R}")
    out()

    prog = step_program()
    onames = [n for _, n in prog.output_ports]
    out(f"  {GREY}compile{R} {CHECK} one generation program "
        f"{DIM}(3x3 conv + rule, compiled once, re-dispatched every step){R}")

    alive = seed()
    frames = [alive[0, 0].copy()]
    ane_t = 0.0
    t0 = time.perf_counter()
    for s in range(STEPS):
        t = time.perf_counter()
        res = prog(alive)
        ane_t += time.perf_counter() - t
        alive = res[onames[0]].reshape(1, 1, N, N)
        if (s + 1) % FRAME_STRIDE == 0:
            frames.append(alive[0, 0].copy())
    wall = time.perf_counter() - t0

    a = alive[0, 0]
    binary = bool(np.isin(np.unique(a), (0.0, 1.0)).all())     # the rule must stay boolean
    pop = float(a.sum())
    energy = ANE_RAIL_W * ane_t

    out(f"  {GREY}evolve{R}  {CHECK} {STEPS} generations on the ANE "
        f"{DIM}({secs(wall)} wall, {ane_t * 1e3 / STEPS:.2f} ms/step){R}")
    out(f"  {GREY}field{R}   {DIM}population {pop:.0f} of {N * N} cells, "
        f"{'boolean' if binary else 'NON-BOOLEAN'}{R}")
    out(f"  {GREY}energy{R}  {DIM}~{secs(ane_t)} of ANE step time at the measured "
        f"~{ANE_RAIL_W} W rail ~ {BOLD}{joules(energy)}{R}{DIM} for the whole run{R}")
    out()

    wrote = render(frames)
    if wrote:
        out(f"  {CHECK} {BOLD}wrote {wrote}{R}")
    prog.release()

    ok = binary and 0.0 < pop < N * N
    out(f"  {('PASS' if ok else 'FAIL')}: the ANE ran {STEPS} Life generations on the "
        f"engine, the field stayed boolean and alive")
    return 0 if ok else 1


# rendering
def render(frames):
    try:
        from PIL import Image
    except ImportError:
        out(f"  {DIM}(install Pillow to write the animation: pip install pillow){R}")
        return None
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ims = []
    for f in frames:
        live = f > 0.5
        rgb = np.empty(f.shape + (3,), np.uint8)
        rgb[~live] = (10, 14, 20)                    # near-black background
        rgb[live] = (72, 187, 170)                   # brand teal
        ims.append(Image.fromarray(rgb).resize((ANIM_SIZE, ANIM_SIZE), Image.NEAREST))
    durs = [ANIM_MS] * len(ims); durs[-1] = ANIM_HOLD
    ims[0].save(assets / "game_of_life.webp", save_all=True, append_images=ims[1:],
                duration=durs, loop=0, quality=70, method=6)
    return f"docs/assets/game_of_life.webp (animated WebP, {len(ims)} frames)"


if __name__ == "__main__":
    sys.exit(main())

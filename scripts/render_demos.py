"""Render animated GIFs (untrained vs trained, side-by-side) per subtask.

For each saved checkpoint, runs both a random-action episode and a
deterministic-policy episode, samples frames at a fixed interval, and
writes a side-by-side animated GIF to `assets/<subtask>_before_after.gif`.

Run from the project root with the venv active:
    python scripts/render_demos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import TQC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402  (path-dependent)

CHECKPOINTS = [
    ("microwave", "iter1_microwave_TQC.zip"),
    ("slide cabinet", "iter2_slide_cabinet_TQC.zip"),
    ("hinge cabinet", "iter3_hinge_cabinet_TQC.zip"),
    ("light switch", "iter4_light_switch_TQC.zip"),
    ("bottom burner", "iter5_bottom_burner_TQC.zip"),
    ("top burner", "iter6_top_burner_TQC.zip"),
    ("kettle", "iter7_kettle_TQC.zip"),
]

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

ROLLOUT_STEPS = 280
FRAME_STRIDE = 7      # sample every Nth env step → ~40 frames / 280-step episode
GIF_FPS = 20
DOWNSCALE = 2         # halve resolution to keep GIF size reasonable


def rollout_frames(env, policy_fn, steps: int = ROLLOUT_STEPS) -> list[np.ndarray]:
    """Run a rollout and return one frame per env step (always `steps` frames).

    If the env terminates early, the final frame is repeated so the two
    rollouts (untrained / trained) line up frame-for-frame in the side-by-side.
    """
    obs, _ = env.reset(seed=0)
    frames = [env.render()]
    done = False
    last = frames[0]
    for _ in range(steps):
        if not done:
            action = policy_fn(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            last = env.render()
            done = bool(terminated or truncated)
        frames.append(last)
    return frames[::FRAME_STRIDE]


def label(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    if DOWNSCALE > 1:
        img = img.resize((img.width // DOWNSCALE, img.height // DOWNSCALE), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 1, bbox[2] + 3, bbox[3] + 1), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def stitch(left: Image.Image, right: Image.Image) -> np.ndarray:
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width + 6, h), color=(20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 6, 0))
    return np.asarray(canvas)


def render_subtask(subtask: str, ckpt: str) -> Path | None:
    ckpt_path = PROJECT_ROOT / ckpt
    if not ckpt_path.exists():
        print(f"  SKIP: checkpoint {ckpt} not found")
        return None

    train.SUBTASK = subtask

    env = train.make_env(render_mode="rgb_array", with_reward_wrapper=False)
    untrained = rollout_frames(env, lambda _o: env.action_space.sample())
    env.close()

    env = train.make_env(render_mode="rgb_array", with_reward_wrapper=False)
    model = TQC.load(str(ckpt_path), env=env, device="cpu")
    trained = rollout_frames(env, lambda o: model.predict(o, deterministic=True)[0])
    env.close()

    n = min(len(untrained), len(trained))
    composed: list[np.ndarray] = []
    for i in range(n):
        l = label(untrained[i], f"{subtask} — untrained")
        r = label(trained[i], f"{subtask} — trained (TQC + HER)")
        composed.append(stitch(l, r))

    out = ASSETS_DIR / f"{subtask.replace(' ', '_')}_before_after.gif"
    imageio.mimsave(out, composed, fps=GIF_FPS, loop=0)
    print(f"  wrote {out.name}  ({len(composed)} frames)")
    return out


def main():
    for subtask, ckpt in CHECKPOINTS:
        print(f"[{subtask}]")
        render_subtask(subtask, ckpt)


if __name__ == "__main__":
    main()

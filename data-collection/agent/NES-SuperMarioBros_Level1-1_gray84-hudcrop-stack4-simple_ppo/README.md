---
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
license: mit
tags:
  - reinforcement-learning
  - stable-baselines3
  - ppo
  - supermariobrosnes-turbo
  - rlab
  - SuperMarioBros-Nes-v0
metrics:
  - success-rate
---

# SuperMarioBros-Nes-v0 — Level1-1 — PPO

Stable-Baselines3 PPO policy for `SuperMarioBros-Nes-v0` `Level1-1`, trained and evaluated with
[`rlab`](https://github.com/tsilva/rlab).

## At a Glance

| Item | Value |
|---|---|
| Task | Complete `SuperMarioBros-Nes-v0` `Level1-1` |
| Provider | `supermariobrosnes-turbo` |
| Algorithm | `ppo` |
| Checkpoint | Step `4000000` |
| Evaluation | `stochastic` full evaluation, `100` episodes |
| Success | minimum `96.0%`, mean `96.0%` |
| Mean return | `3049.776` |
| Release | `v1` |
| Preview | Root `replay.mp4` |
| YouTube | [Watch on YouTube](https://www.youtube.com/watch?v=LQ4x1Sr5TSI) |

## Quick Start

```bash
git clone https://github.com/tsilva/rlab
cd rlab
git checkout 044dcc3d2a978fb1a77524822238c4c7814c002d
uv sync --frozen
```

Import the ROM, then play or evaluate the immutable checkpoint:

```bash
uv run rlab import-roms ~/roms --game SuperMarioBros-Nes-v0
uv run rlab play https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo/resolve/v1/model.zip
uv run rlab eval https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo/resolve/v1/model.zip
```

## Evaluation

Action selection was `stochastic` under the published evaluation environment contract.

| Start | Episodes | Successes | Success rate | Mean return |
|---|---:|---:|---:|---:|
| Level1-1 | 100 | 96 | 96.0% | 3049.776 |

## Environment and Policy Contract

| Item | Value |
|---|---|
| Environment | `supermariobrosnes-turbo:SuperMarioBros-Nes-v0` |
| Environment hash | `sha256:557f6d534a3d2edc8eebe16a5fc97d39de9776ffe9d4fc6e4bdde20f0a0d150b` |
| Preprocessing | `{"frame_skip":4,"frame_stack":4,"max_pool_frames":false,"obs_copy":"safe_view","obs_crop":[32,0,0,0],"obs_crop_fill":0,"obs_crop_mode":"remove","obs_grayscale":true,"obs_resize":[84,84],"obs_resize_algorithm":"area","pipeline":"supermariobrosnes_turbo_native_vec_env","policy_observation_layout":"channel_first","sticky_action_prob":0.0}` |
| Action contract | `{"set":"simple"}` |

## Provenance

| Item | Value |
|---|---|
| Source | [rlab](https://github.com/tsilva/rlab) |
| Run | [Level1-1_base_s7_20260704T204541Z](https://wandb.ai/tsilva/SuperMarioBros-Nes-v0/runs/vnj2jxi5) |
| Recipe | `base` |
| Seed | `7` |
| Source commit | `044dcc3d2a978fb1a77524822238c4c7814c002d` |
| Evaluated artifact | `tsilva/SuperMarioBros-Nes-v0/Level1-1_base_s7_20260704T204541Z-checkpoint:step-4000000` |

## Files

| File | Purpose |
|---|---|
| `model.zip` | Stable-Baselines3 policy checkpoint |
| `model.json` | Versioned checkpoint identity, policy type, provenance, and recipe binding |
| `recipe.json` | Versioned execution and evaluation contract |
| `release_manifest.json` | Release identity, evaluation evidence, and artifact hashes |
| `replay.mp4` | Browser-safe representative episode |
| `LICENSE` | License for rlab-authored policy weights and publication material |

## Limitations

Evaluation establishes performance only for the published environment hash, start distribution,
policy preprocessing, and action-selection protocol. It does not establish generalization to
other levels, environments, ROM revisions, or contracts.

## Licensing

The rlab-authored policy weights and publication material are licensed under the MIT License in
`LICENSE`. Emulator/runtime software and game assets remain governed by their own licenses and
terms. This repository does not redistribute a game ROM.

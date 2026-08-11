# mario-world

A local experiment in building a small GameNGen-style world model for **Super Mario Bros. (1986), level 1-1**.

The goal is to learn a playable neural simulator that predicts the next game frame from recent gameplay frames and controller inputs. 

## Teacher Forcing vs Autoregressive Play

During training, the model uses **teacher forcing**:

```text
real previous frames + recorded actions -> predict real next frame
```

During inference/play, the model runs **autoregressively**:

```text
generated previous frames + live player action -> generate next frame
```

This creates a distribution shift. The model trains on clean emulator frames, but at inference it must condition on its own imperfect generated frames. The GameNGen paper mitigates this with **noise augmentation**: during training, it adds noise to the previous-frame latents so the model learns to recover from slightly corrupted history.

## Data Plan

The first dataset should come from emulator rollouts of level 1-1. A pretrained or scripted agent (https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo) can play the game while the collector records:

```text
frames:  uint8 [T, H, W, 3]
actions: int   [T]
dones:   bool  [T]
info:    optional metadata such as x position, score, status
```

Mario 1-1 is much simpler than DOOM, but coverage still matters. The data should include walking, running, jumping, enemies, pipes, blocks, coins, death states, and the flagpole. A purely random policy is likely too weak because it may die early or fail to explore the level, so a scripted/noisy policy or pretrained agent is preferred.

## Data Collection Strategy

The current plan is to start from a strong pretrained level 1-1 agent, then deliberately perturb it to collect a wider range of gameplay states.

The two preferred diversity mechanisms are:

- **Neuron perturbation:** inject stochasticity into the trained agent by enabling dropout-like behavior or adding noise to hidden activations/logits. The goal is to produce competent-but-imperfect play: missed jumps, hesitation, alternate timings, collisions, and deaths while still remaining recognizably Mario-like.
- **Start-state randomization:** save emulator states at many positions throughout level 1-1, then begin rollouts from those states instead of always starting at the beginning. This is likely the most important coverage tool because it prevents the dataset from being dominated by early-level states.

## Workflow Style

Keep the project simple, modular, and idempotent. Each stage should be easy to rerun without surprising side effects.

Preferred shape:

```text
data-collection/
  raw_episodes.ipynb
training/
  encode_frames.ipynb
  train_world_model.ipynb
  agent/
  data/
  models/
inference/
  run_inference.ipynb
```

Raw collection should not encode episodes using the VAE.

### Setup Requirements & Artifacts

- **ROM Path:** `training/agent/Rom/SuperMarioBros` (registered with `stable_retro` as `SuperMarioBros-Nes-v0`).
- **Model Checkpoint:** `training/agent/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo/model.zip`.
- **Contract Metadata:** `model.metadata.json` sidecar configures:
  - Provider: `supermariobrosnes-turbo`
  - HUD Crop: `hud_crop_top: 32` (`obs_crop: [32, 0, 0, 0]`)
  - Action Set: `simple`
  - Episode Max Steps: `4500`

### Interactive Playback (GUI Window)

To watch the agent play in real-time in a GUI window:

```bash
uv run --directory training/agent/rlab rlab play \
  --game SuperMarioBros-Nes-v0 \
  --model ../NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo/model.zip \
  --scale 4
```

> **Note on FPS & Engine:** Omit `--fps` throttling to allow the native `supermariobrosnes-turbo` engine to step frame-accurately without Pygame timer sleep stutters. Including the fps makes the model and the gameplay it sees out of sync leading to poor performance.

This could be an additional pertrubance mechanism (tbd!)

### Headless Evaluation

To evaluate policy rollouts in the terminal:

```bash
uv run --directory training/agent/rlab rlab eval \
  --game SuperMarioBros-Nes-v0 \
  --model ../NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo/model.zip \
  --episodes 50
```

## Open Implementation Areas

- [x] Emulator/sandbox setup & pretrained agent inference
- [ ] Create dataset
    - [ ] Draft raw episode collection notebook
    - [ ] Collect 10 first-training episodes
- [ ] Create VAE/frame latent encoding
- [ ] Create encoded dataset
- [ ] Action-conditioned diffusion model
- [ ] Autoregressive inference loop
- [ ] Evaluation and visual debugging

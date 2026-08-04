# mario-world

A local experiment in building a small GameNGen-style world model for **Super Mario Bros. (1986), level 1-1**.

The goal is to learn a playable neural simulator that predicts the next game frame from recent gameplay frames and controller inputs. This repo is intentionally starting small: first the background and data pipeline, then model training and inference as the implementation takes shape.

## Background

This project is inspired by the GameNGen paper, *Diffusion Models Are Real-Time Game Engines*. The key idea is to treat a game as an interactive environment:

```text
latent game state + player action -> next latent game state -> rendered pixels
```

A traditional emulator has access to the real game state and rendering code. A learned world model does not. It only sees observations, usually screen pixels, plus the actions taken by the player or agent. The model is trained to approximate:

```text
q(next_frame | previous_frames, actions)
```

In this sense, the model is more than a normal video generator. It is still a diffusion model architecturally, but it is used as an action-conditioned simulator in a closed loop: each generated frame becomes part of the context for generating the next frame.

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

## Conditioning

The GameNGen approach repurposes a pretrained text-to-image diffusion model:

- The target next frame is encoded into a latent and denoised by the diffusion U-Net.
- Previous frames are encoded with the VAE and concatenated as extra visual context.
- Text conditioning is removed.
- Controller inputs are embedded as learned action tokens and passed through the conditioning path that would normally attend to text.

For Mario, the equivalent training sample is:

```text
last N frames + last N actions -> next frame
```

The paper used a context length of 64 frames/actions for DOOM, or a little over 3 seconds at 20 FPS. For this project, a smaller context such as 16 or 32 frames is a reasonable first target, with 64 as a later experiment.

## Data Plan

The first dataset should come from emulator rollouts of level 1-1. A pretrained or scripted agent can play the game while the collector records:

```text
frames:  uint8 [T, H, W, 3]
actions: int   [T]
dones:   bool  [T]
info:    optional metadata such as x position, score, status
```

The training loader will sample windows from these episodes:

```text
frames[t-N:t], actions[t-N:t] -> frame[t]
```

For a first local prototype, the rough dataset ladder is:

```text
100k examples: pipeline smoke test
1M examples: first real model
5M examples: stronger prototype
10M+ examples: only if quality is still improving
```

Mario 1-1 is much simpler than DOOM, but coverage still matters. The data should include walking, running, jumping, enemies, pipes, blocks, coins, death states, and the flagpole. A purely random policy is likely too weak because it may die early or fail to explore the level, so a scripted/noisy policy or pretrained agent is preferred.

## Local Constraints

This project is intended to run locally on a MacBook Pro M3. That changes the design target:

- Start with lower resolution, likely `128x128` or native NES-scale preprocessing.
- Prefer a small latent diffusion model before attempting full Stable Diffusion fine-tuning.
- Aim for 1-4 denoising steps at inference if interactive speed is required.
- Treat 20 FPS as an eventual target, not the first milestone.

## Open Implementation Areas

- Emulator/sandbox setup
- Rollout collection
- Dataset format and dataloader
- VAE/frame latent encoding
- Action-conditioned diffusion model
- Autoregressive inference loop
- Evaluation and visual debugging

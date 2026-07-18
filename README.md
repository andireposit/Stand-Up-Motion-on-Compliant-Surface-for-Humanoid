# Demonstration-Guided Humanoid Stand-Up on an Emulated Deformable Surface

This repository contains the evaluation code, trained policy, MuJoCo model, motion reference, and supplementary media for the research work:

> **Demonstration-Guided Humanoid Stand-Up on an Emulated Deformable Surface**  
> Aniruddh Kushwah, Vyankatesh Ashtekar, and Ashish Dutta  
> Department of Mechanical Engineering, Indian Institute of Technology Kanpur

The project studies reference-guided reinforcement learning for generating a fallen-to-standing motion for the 29-DOF Unitree G1 humanoid on compliant soft ground. A human stand-up demonstration recorded on hard ground is used as the motion reference, and the learned policy is subsequently adapted to a softer MuJoCo contact model.

## Overview

Standing up from a fallen configuration is a contact-rich, non-cyclic whole-body task. The problem becomes more difficult on compliant terrain because the ground must deform before sufficient support forces are generated.

The method implemented in this project combines:

- A retargeted human stand-up demonstration
- Residual joint-position control
- Proximal Policy Optimization (PPO)
- Reference-motion tracking rewards
- Explicit recovery rewards for pelvis height, torso uprightness, and final standing posture
- MuJoCo soft-contact parameters for emulating compliant terrain
- A two-stage hard-ground and soft-ground training procedure

The released evaluation script loads one trained policy and reproduces the stand-up behavior in the MuJoCo viewer.

## Stand-up sequence

![Sequential snapshots of the learned stand-up policy](media/figure2.png)

**Figure 2.** Sequential snapshots of the Unitree G1 stand-up policy in the MuJoCo simulation environment.

## Supplementary video

[▶ Watch the stand-up policy evaluation video](https://youtu.be/c04fnMCDdd8?si=WEkKwNhVSA3PFn5V)

## Repository structure

```text
.
├── assets/
│   ├── *.STL
│   └── ...other Unitree G1 mesh assets
│
├── media/
│   ├── figure2.png
│   └── standup_policy.mp4
│
├── model/
│   ├── g1_fallen_stand_soft_ground.zip
│   └── g1_fallen_stand_soft_ground_vecnormalize.pkl
│
├── .gitattributes
├── LICENSE
├── README.md
├── eval.py
├── g1.xml
├── requirements.txt
└── stand_up_lying_R_002__A475_new.csv
```

## Method

### Reference-guided reinforcement learning

The policy operates at 60 Hz, while the reference motion is sampled at 120 Hz. At each control step, the policy predicts a 29-dimensional residual action that modifies the reference joint positions:

```text
commanded joint position = reference joint position + action scale × policy action
```

The action scale used in the evaluation environment is:

```text
0.25
```

The commanded joint positions are tracked by joint-level proportional-derivative controllers defined in the MuJoCo model.

### Policy observation

The policy observation contains:

- Current simulated robot state
- Current reference state
- Joint-position tracking error
- Joint-velocity tracking error
- Deviation from the final standing posture
- Normalized motion phase
- Previous action

The normalized phase informs the policy about its progress through the stand-up sequence.

### Reward design

The complete reward combines three groups of objectives.

#### Reference tracking

- Joint-pose tracking
- Joint-velocity tracking
- Root planar-position tracking
- Pelvis-height tracking
- Root-orientation tracking

#### Final recovery

- Final standing-pose tracking
- Torso uprightness
- Standing pelvis height

#### Regularization

- Foot-slip penalty
- Action-magnitude penalty
- Action-smoothness penalty

A phase-dependent weighting gradually shifts the objective from reference tracking toward completion of the final standing posture.

### Two-stage training

The policy is trained in two stages:

1. **Hard-ground training**  
   The policy first learns to track the demonstrated motion and complete the stand-up task using the default MuJoCo contact settings.

2. **Compliant-ground fine-tuning**  
   Training is continued after reducing contact stiffness and expanding the nominal penetration region through the MuJoCo `solref` and `solimp` parameters.

The compliant-ground parameters reported in the paper are:

```text
solref = (0.1, 1)
solimp = (0.0, 0.95, 0.02)
```

These parameters emulate delayed support-force generation and substantial contact penetration while retaining a planar floor geometry.

## Main results

For the representative policy reported in the paper:

| Metric | Result |
|---|---:|
| Final pelvis height | 0.792 m |
| Target pelvis height | 0.794 m |
| Final uprightness | 0.991 |
| Maximum uprightness | 1.000 |
| Maximum contact penetration | Approximately 40 mm |

The learned policy preserved the overall demonstrated stand-up sequence while making small joint-space adaptations during contact-intensive phases.

The reward ablation showed that reference tracking alone was insufficient for successful recovery. Although the tracking-only and complete-reward policies had similar aggregate joint-position errors, only the complete reward produced a stable standing configuration.

| Metric | Tracking only | Complete reward |
|---|---:|---:|
| Aggregate RMS joint error | 0.1571 rad | 0.1560 rad |
| Final pelvis height | 0.059 m | 0.792 m |
| Maximum pelvis height | 0.130 m | 0.794 m |
| Final uprightness | 0.543 | 0.991 |
| Maximum uprightness | 0.917 | 1.000 |

These results indicate that accurate joint-space imitation does not by itself guarantee successful whole-body recovery under compliant contact.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/andireposit/Stand-Up-Motion-on-Compliant-Surface-for-Humanoid.git
cd Stand-Up-Motion-on-Compliant-Surface-for-Humanoid
```

### 2. Download Git LFS files

The trained policy, normalization statistics, mesh files, and video may be tracked with Git LFS.

```bash
git lfs install
git lfs pull
```

### 3. Create a virtual environment

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the MuJoCo evaluation

Run the following command from the repository root:

```bash
python eval.py
```

The script will:

1. Load the combined Unitree G1 model and scene from `g1.xml`.
2. Load the retargeted motion reference from the CSV file.
3. Create the custom Gymnasium evaluation environment.
4. Restore the saved `VecNormalize` statistics.
5. Load the trained PPO policy.
6. Run deterministic policy inference.
7. Display the motion in the MuJoCo passive viewer.

Close the viewer window to stop the evaluation.

## Expected paths

The paths in `eval.py` should match the repository layout:

```python
XML_PATH = "g1.xml"
CSV_PATH = "stand_up_lying_R_002__A475_new.csv"

MODEL_PATH = (
    "model/"
    "g1_fallen_stand_soft_ground_74000000_steps.zip"
)

VECNORM_PATH = (
    "model/"
    "g1_fallen_stand_soft_ground_vecnormalize_74000000_steps.pkl"
)
```

Update the model file names when different checkpoints are used.

## Default evaluation conditions

The provided visualization uses deterministic and noise-free evaluation:

```python
reset_joint_noise = 0.0
reset_vel_noise = 0.0
reset_root_xy_noise = 0.0
reset_yaw_noise = 0.0

push_prob = 0.0
push_force_range = (0.0, 0.0)
push_duration_range = (0, 0)

friction_range = (0.6, 0.6)
gravity_z_range = (-9.81, -9.81)
```

These settings are intended to reproduce a clean policy rollout. They do not represent the full reset-noise curriculum used during compliant-ground fine-tuning.

## Motion-reference data

The motion reference was obtained from a human stand-up sequence and retargeted to the Unitree G1 embodiment.

The CSV contains:

- Root translation
- Root orientation
- Twenty-nine actuated joint angles

Required root columns include:

```text
root_translateX
root_translateY
root_translateZ
root_rotateX
root_rotateY
root_rotateZ
```

Joint angles are interpreted as degrees and converted to radians by the evaluation environment. Reference velocities are estimated using finite differences.

## MuJoCo model and assets

The root-level `g1.xml` contains both the Unitree G1 robot definition and the simulation scene.

The mesh assets are stored in:

```text
assets/
```

The XML and assets must preserve the same body, joint, actuator, sensor, and keyframe ordering used during training.

Important keyframes include:

```text
fallen
stand
```

Important body names include:

```text
pelvis
left_ankle_roll_link
right_ankle_roll_link
```

## Model compatibility

The following files must remain mutually compatible:

- `eval.py`
- `g1.xml`
- Motion-reference CSV
- PPO checkpoint
- `VecNormalize` statistics

Changing the observation layout, actuator ordering, robot model, action scale, control timestep, or normalization file may cause shape errors or incorrect behavior.

The PPO checkpoint and `VecNormalize` file must originate from the same training run.

## Troubleshooting

### A file cannot be found

Run the command from the repository root:

```bash
python eval.py
```

Then verify that all paths in `eval.py` match the repository file names.

### MuJoCo cannot load a mesh

Check that:

- `assets/` is beside `g1.xml`.
- Every referenced mesh is present.
- File-name capitalization matches the XML.
- Git LFS content has been downloaded.

### The policy or normalization file cannot be loaded

Confirm that the policy `.zip` and normalization `.pkl` files are both present in `model/`.

### Observation-space or tensor-shape error

This normally indicates an incompatibility between the trained policy and the current environment definition. Restore the environment, XML, motion CSV, checkpoint, and normalization file from the same experiment.

### The viewer does not open

The passive MuJoCo viewer requires a graphical desktop and working OpenGL support. Headless servers, containers, SSH sessions, and some WSL installations may require additional display configuration.

### The robot does not stand correctly

Verify that:

- The intended PPO checkpoint is loaded.
- The matching normalization file is loaded.
- `VecNormalize.training` is `False`.
- Reward normalization is disabled during evaluation.
- The correct motion CSV is used.
- The XML and mesh assets match training.
- The action scale is unchanged.

## Limitations

The compliant surface is emulated using MuJoCo's soft-contact formulation rather than an explicit deformable-material model.

The model primarily changes normal contact behavior. It does not fully reproduce:

- Tangential terrain deformation
- Foot submergence resistance
- Material hysteresis
- Spatially varying compliance
- Permanent terrain deformation

The current results are limited to simulation. Physical validation and improved deformable-terrain modeling remain future work.

## Citation

A BibTeX entry can be added or updated after the final publication details are available:

## Git LFS

Recommended tracking rules:

```bash
git lfs install
git lfs track "*.zip"
git lfs track "*.pkl"
git lfs track "*.STL"
git lfs track "*.stl"
git lfs track "*.mp4"
git add .gitattributes
```

Commit the new media and README:

```bash
git add README.md media/figure2.png media/standup_policy.mp4
git commit -m "Update README with paper details and supplementary media"
git push
```

## License

See [`LICENSE`](LICENSE).

Before redistributing the repository, confirm that you have permission to share:

- Unitree G1 model and mesh assets
- Retargeted motion-reference data
- Trained policy checkpoint
- Saved normalization statistics
- Paper figures and supplementary video

## Acknowledgements

This project uses:

- [MuJoCo](https://mujoco.org/) for physics simulation and visualization
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO
- [Gymnasium](https://gymnasium.farama.org/) for the reinforcement-learning environment interface

The stand-up demonstrations are based on motions from the BONES-SEED dataset and were retargeted to the Unitree G1 using a kinematic retargeting pipeline.

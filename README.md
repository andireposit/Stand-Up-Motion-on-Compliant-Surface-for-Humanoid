# Unitree G1 Fallen-to-Stand Policy Evaluation on Compliant Surface

This repository provides a complete evaluation and visualization setup for a trained **Unitree G1 fallen-to-stand policy**.

Running `eval.py` loads the trained PPO policy and its matching `VecNormalize` statistics, creates the custom MuJoCo environment, and displays the robot standing up in the MuJoCo viewer.

## Repository structure

```text
.
├── assets/
│   ├── *.STL
│   └── ...other MuJoCo mesh assets
│
├── model/
│   ├── g1_fallen_stand_soft_ground_74000000_steps.zip
│   └── g1_fallen_stand_soft_ground_vecnormalize_74000000_steps.pkl
│
├── .gitattributes
├── LICENSE
├── README.md
├── eval.py
├── g1.xml
├── requirements.txt
└── stand_up_lying_R_002__A475_new.csv
```

### Main files

| File or directory | Description |
|---|---|
| `eval.py` | Contains the custom Gymnasium environment and the policy visualization loop. |
| `g1.xml` | Combined MuJoCo robot model and scene definition. |
| `assets/` | Mesh files referenced by `g1.xml`. |
| `model/` | Trained PPO policy and matching `VecNormalize` statistics. |
| `stand_up_lying_R_002__A475_new.csv` | Reference stand-up motion used by the environment. |
| `requirements.txt` | Python dependencies required to run the evaluation. |

## What the evaluation does

The evaluation script:

1. Loads the Unitree G1 model and scene from `g1.xml`.
2. Loads the stand-up reference trajectory from the CSV file.
3. Creates the custom `G1FallenStandEnv` Gymnasium environment.
4. Restores the saved `VecNormalize` observation statistics.
5. Loads the trained Stable-Baselines3 PPO policy.
6. Runs deterministic policy inference.
7. Displays the motion in the MuJoCo passive viewer.
8. Resets the environment automatically when an episode finishes.

The default visualization uses clean evaluation conditions:

- No joint reset noise
- No velocity reset noise
- No root-position or yaw reset noise
- No external pushes
- Fixed ground friction
- Standard gravity
- Deterministic policy actions

## Requirements

- Python 3.10 or newer
- A desktop environment with OpenGL support
- Git LFS when the model or asset files are stored using LFS

The main Python packages are:

- MuJoCo
- Stable-Baselines3
- Gymnasium
- PyTorch
- NumPy
- pandas
- SciPy

## Installation

Clone the repository:

```bash
git clone https://github.com/andireposit/Stand-Up-Motion-on-Compliant-Surface-for-Humanoid.git
cd Stand-Up-Motion-on-Compliant-Surface-for-Humanoid
```

When the repository uses Git LFS, download the tracked files:

```bash
git lfs install
git lfs pull
```

Create a virtual environment.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the visualization

Run the script from the repository root:

```bash
python eval.py
```

The MuJoCo viewer should open and display the trained Unitree G1 stand-up policy.

Close the viewer window to stop the program.

## Expected paths in `eval.py`

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

When your model files have different names, update `MODEL_PATH` and `VECNORM_PATH` accordingly.

## Console output

During evaluation, the script periodically prints information such as:

```text
step=0030 reward=... height=... target_height=...
upright=... phase=... standing_weight=...
```

The values indicate:

- `step`: current control step
- `reward`: accumulated episode reward
- `height`: current root or pelvis height
- `target_height`: reference-motion height
- `upright`: upright-orientation reward
- `phase`: progress through the reference motion
- `standing_weight`: transition weight toward the final standing objective

## Important compatibility notes

The policy file and `VecNormalize` file must come from the **same training run**.

The trained policy also depends on the environment configuration used during training. Changing the following can cause loading errors or poor behavior:

- Observation ordering or dimensions
- Action dimensions
- Joint ordering
- Actuator ordering
- Robot model
- Reference-motion columns
- Action scale
- Control timestep
- Normalization statistics

The current `eval.py`, `g1.xml`, CSV file, policy checkpoint, and normalization file should therefore be kept together.

## Motion-reference CSV

The motion CSV supplies the root pose and 29 joint-angle trajectories used as the reference motion.

Required root columns include:

```text
root_translateX
root_translateY
root_translateZ
root_rotateX
root_rotateY
root_rotateZ
```

The joint values are read as degrees and converted to radians by `eval.py`.

## MuJoCo assets

`g1.xml` uses:

```xml
<compiler angle="radian" meshdir="assets"/>
```

Therefore, the `assets` directory must remain next to `g1.xml`.

Do not rename or move mesh files unless the corresponding paths in `g1.xml` are also updated. File-name capitalization matters on Linux.

## Troubleshooting

### `FileNotFoundError`

Make sure the command is being run from the repository root:

```bash
python eval.py
```

Also verify that the paths configured in `eval.py` match the file names in the repository.

### MuJoCo cannot load a mesh

Confirm that:

- The `assets/` directory exists beside `g1.xml`.
- All mesh files referenced by `g1.xml` are present.
- File names and capitalization match the XML exactly.
- Git LFS files were downloaded with `git lfs pull`.

### Model file cannot be loaded

Check that both files exist in `model/`:

```text
trained_policy.zip
matching_vecnormalize.pkl
```

The `.zip` extension should not be removed from a Stable-Baselines3 model checkpoint.

### Observation-space or tensor-shape error

This usually means that the policy or `VecNormalize` file was produced using a different environment definition.

Use the policy, normalization statistics, environment code, robot XML, and CSV from the same experiment.

### Viewer does not open

The MuJoCo viewer requires a graphical desktop and working OpenGL support.

Running through a headless server, container, SSH session, or WSL installation may require additional display configuration.

### The robot behaves incorrectly

Verify that:

- The correct PPO checkpoint is loaded.
- The matching `VecNormalize` file is loaded.
- `VecNormalize.training` is set to `False`.
- Reward normalization is disabled during evaluation.
- The correct motion CSV is used.
- The XML and assets match the training setup.
- Evaluation noise and external pushes are disabled for the initial test.

## Git LFS

Model checkpoints and mesh assets may be too large for normal Git storage. GitHub rejects individual files larger than 100 MB.

Example Git LFS configuration:

```bash
git lfs install
git lfs track "*.zip"
git lfs track "*.pkl"
git lfs track "*.STL"
git lfs track "*.stl"
git add .gitattributes
```

Commit and push the repository:

```bash
git add .
git commit -m "Add Unitree G1 policy evaluation"
git push
```

## License

See the [`LICENSE`](LICENSE) file for the repository license.

Before redistributing this repository, ensure that you have permission to share:

- The Unitree G1 robot model and mesh assets
- The reference-motion CSV
- The trained policy
- The saved normalization statistics

## Acknowledgements

This project uses:

- [MuJoCo](https://mujoco.org/) for physics simulation and visualization
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO inference
- [Gymnasium](https://gymnasium.farama.org/) for the reinforcement-learning environment interface

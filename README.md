# TrafficGAN+

Code for the paper **"Temporal Gradient Penalty and Attack-Scenario Entropy for Sequence GANs in Intrusion Detection: An Empirical Evaluation of TrafficGAN+"** by Zaid Arafat and Mikhail Ronkin.

TrafficGAN+ is a conditional CNN-LSTM Wasserstein GAN with gradient penalty (WGAN-GP) that generates windows of consecutive network flows for intrusion-detection training. The package implements two regularizers and evaluates them against a wide set of baselines:

- **Temporal gradient penalty (TGP):** penalizes the critic gradient with respect to each inter-flow increment (Eqs. 4–6 of the paper).
- **Attack scenario entropy (ASE):** maximizes the entropy of the batch-marginal class distribution assigned to generated windows by an auxiliary classifier trained on real traffic (Eqs. 9–11).

The package reproduces every table and figure in Section 4 of the paper from the raw dataset files. The main finding is negative: neither regularizer improves significantly on the same backbone without it. A TabDDPM-style diffusion model and SMOTE are the strongest methods in the evaluation.

## Installation

Python 3.10–3.12 and an NVIDIA GPU are recommended.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt
python tests/test_tgp.py             # checks Eq. (5) and Proposition 1 numerically
```

## Data

Download the original CSV files and place them in the folders below, or edit `paths.datasets` in `configs/default.yaml`. Sub-folders are searched.

| Folder | Dataset | Files used |
|---|---|---|
| `data/raw/cicids2017/` | [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) | the 8 MachineLearningCVE CSVs (`*.pcap_ISCX.csv`) |
| `data/raw/unsw_nb15/` | [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | the raw `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` |
| `data/raw/cicids2018/` | [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | the 10 daily CSVs (*Processed Traffic Data for ML Algorithms*) |

Processed data are written to `data/processed/` and results to `results/`. About 10 GB of free disk space is needed.

## Running the experiments

Quick check on tiny synthetic data (a few minutes; the numbers are meaningless):

```bash
bash tests/smoke_test.sh        # Windows: tests\smoke_test.bat
```

Full schedule:

```bash
bash scripts/run_all.sh         # Windows: scripts\run_all.bat
```

Or stage by stage. Every stage is resumable: runs that have already finished are skipped.

```bash
python scripts/run_experiments.py prepare     --dataset cicids2017
python scripts/run_experiments.py main        --dataset cicids2017
python scripts/run_experiments.py sensitivity --dataset cicids2017
python scripts/run_experiments.py prepare     --dataset unsw_nb15
python scripts/run_experiments.py main        --dataset unsw_nb15  --methods none smote ctgan timegan dgan wgan_gp trafficgan
python scripts/run_experiments.py prepare     --dataset cicids2018
python scripts/run_experiments.py main        --dataset cicids2018 --methods none smote ctgan timegan dgan wgan_gp trafficgan
python scripts/run_experiments.py transfer    --source cicids2017 --target cicids2018
python scripts/make_tables.py
python scripts/export_for_paper.py
```

Configuration values can be overridden from the command line, for example `--set train.n_iters=5000`.

In the paper, CTGAN, CopulaGAN, DoppelGANger and the four alternative regularizers use five seeds (0–4), the sensitivity study uses three, and all other methods use ten. `scripts/run_rest.bat` and `scripts/rerun_v2*.bat` are the Windows scripts used for this schedule.

The full schedule takes several days on one RTX 3090. The critic LSTM runs without cuDNN, because cuDNN cannot double-differentiate an LSTM, so several processes can share one GPU.

## Methods

| Key | Name in the paper | Implementation |
|---|---|---|
| `none`, `ros`, `smote` | No augmentation, ROS, SMOTE | `tg/ids.py` (imbalanced-learn SMOTE, k = 5) |
| `mlp_gan`, `mlp_wgan` | MLP-GAN, MLP WGAN-GP | `tg/train_gan.py`; flow-level, conditional |
| `ctgan`, `copulagan` | CTGAN, CopulaGAN | SDV library; flow-level, conditional sampling |
| `tabddpm` | TabDDPM-style | `tg/baselines/ddpm.py`; Gaussian diffusion with the TabDDPM MLP denoiser |
| `timegan` | TimeGAN | `tg/baselines/timegan.py`; PyTorch port, one model per class |
| `dgan` | DoppelGANger | gretel-synthetics DGAN, one model per class |
| `wgan_gp` | CNN-LSTM WGAN-GP | TrafficGAN+ with γ = β = 0 |
| `trafficgan` | TrafficGAN+ | γ = 5, β = 0.1 |
| `no_tgp`, `no_ase` | Ablations | γ = 0 / β = 0 |
| `alt_sn`, `alt_lp` | TGP → spectral normalization / one-sided GP (WGAN-LP) | GP kept, TGP removed |
| `alt_acgan`, `alt_pacgan` | ASE → AC-GAN class loss / PacGAN | ASE removed |

The critic is class-conditioned through a projection term (Miyato and Koyama, 2018). Flow-level generators produce single flows, and groups of T consecutive generated flows form a window.

## Protocol

- Non-overlapping windows of T = 10 consecutive flows of one class; at most 40,000 windows per class; 70/15/15 split stratified at window level, each partition capped at 20,000 windows per class.
- Identifier and constant columns removed; ±inf replaced by the column's finite max/min and NaN by 0; signed log1p transform of continuous columns and min–max scaling to [−1, 1] fitted on the training partition only; discrete columns snapped to valid values after generation.
- GANs are trained on the training partition with class-balanced real batches; checkpoints are selected by mean KS on the validation partition, with early stopping after 12 checks without improvement.
- Augmentation adds twice the number of training windows for every attack class, to the training partition only. The test partition is never used for training, model selection or augmentation.
- Metrics: KS/TVD per feature, increment-KS, lag-1 ACF error, LSTM discriminative score, precision/recall/density/coverage, class entropy and label agreement from an independent random forest, TSTR, RF and DNN utility, and robustness to ±5% timing perturbation.
- Statistics: mean ± SD over seeds, paired t-test with Holm correction, Wilcoxon signed-rank test, Cohen's d_z and the 95% CI of the paired difference.

## Repository layout

```
configs/default.yaml        all hyperparameters reported in the paper
tg/data.py                  loading, windowing, splitting, scaling
tg/models.py, tg/losses.py  generator, conditional critic, GP, TGP (Eqs. 5–6), ASE (Eqs. 10–11)
tg/train_gan.py             Algorithm 1 and its variants
tg/metrics.py, tg/ids.py    evaluation metrics and IDS classifiers
tg/experiment.py            one run: train → metrics → augmentation → IDS; transfer experiment
tg/baselines/               TabDDPM-style model, TimeGAN, wrappers for SDV and gretel-synthetics
scripts/                    command-line interface, table and figure generation
tests/                      TGP identity test and smoke test on synthetic data
```

## Citation

If you use this code, please cite:

```
Z. Arafat and M. Ronkin, "Temporal Gradient Penalty and Attack-Scenario Entropy for Sequence GANs
in Intrusion Detection: An Empirical Evaluation of TrafficGAN+," manuscript under review, 2026.
```

## License

MIT License. See [LICENSE](LICENSE).

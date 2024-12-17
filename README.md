# Generative Modelling - OMR

# Setup
Install [uv](https://github.com/astral-sh/uv) (fast pip alternative)

On macOS or Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with brew
brew install uv
```

On Windows:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Download dataset
curl https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz --output mnist.npz
# Run vae
uv run train_vae.py # see Usage
```

## Usage
Train a Variational Autoencoder:
```bash
# Basic training
uv run train_vae.py

# Configure hyperparameters
uv run train_vae.py --latent-dim 32 --learning-rate 0.001 --batch-size 64 --num-epochs 50 --batches-per-visual 20 --checkpoint-every 1 --seed 42

# Generate samples from trained model
uv run gen_vae.py --model-path models/vae_final_latent20_lr0.001_batch128.eqx

# Generate interpolations between random points
uv run gen_vae.py --model-path models/vae_final_latent20_lr0.001_batch128.eqx --mode interpolate
```

## Investigation ideas

- K-means clustering on MNIST
  - theoretically talk about measuring difficult by how multi-modal
  - also by how close the clusters are
- PCA for visualising clusters of latent space that correspond to images
- NFs might converge on one mode
- Mode collapse

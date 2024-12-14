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
uv run vae.py --num-epochs 2
```

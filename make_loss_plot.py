'''
A script used one time to make a plot combining batch loss and validation loss
'''

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import wandb


def download_wandb_data(run_id, project_name, entity=None, cache_file="wandb_data.csv"):
    """Download data from W&B and cache it locally"""
    if Path(cache_file).exists():
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)

    # Initialize wandb API
    api = wandb.Api()

    # Get the run
    run_path = f"{entity}/{project_name}/{run_id}" if entity else f"{project_name}/{run_id}"
    print(f"Attempting to access run: {run_path}")

    run = api.run(run_path)
    print(f"Run name: {run.name}")
    print(f"Run ID: {run.id}")

    # Get all history without filtering keys
    print("Downloading history...")
    history = list(run.scan_history())
    print(f"Downloaded {len(history)} records")

    # Convert to DataFrame and save
    df = pd.DataFrame(history)
    df.to_csv(cache_file, index=False)
    print(f"Saved data to {cache_file}")

    return df

def plot_losses(df, output_file="losses.png"):
    plt.figure(figsize=(12, 6))

    # Plot batch losses as a thin line
    if 'training/batch_loss' in df.columns:
        batch_df = df[df['training/batch_loss'].notna()].copy()
        plt.plot(batch_df['training/epoch'], batch_df['training/batch_loss'],
                'b-', label='Batch Loss', alpha=0.3, linewidth=1)

        # Plot smoothed batch losses using rolling mean
        window_size = 100  # Adjust this value to change smoothing
        batch_df['smoothed_loss'] = batch_df['training/batch_loss'].rolling(window=window_size).mean()
        plt.plot(batch_df['training/epoch'], batch_df['smoothed_loss'],
                'b-', label='Batch Loss (smoothed)', linewidth=1)

    # Plot validation losses
    if 'epoch/val_loss' in df.columns:
        val_df = df[df['epoch/val_loss'].notna()].copy()
        plt.plot(val_df['epoch'], val_df['epoch/val_loss'], 'r-',
                label='Validation Loss', linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE MNIST Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save and show the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# Usage example:
if __name__ == "__main__":
    RUN_ID = "c1uzetyo"  # Replace with your run ID
    PROJECT_NAME = "vae-mnist"  # Replace with your project name
    ENTITY = "blinkybool-utrecht-university"  # Replace with your entity name if needed

    # Download or load cached data
    df = download_wandb_data(RUN_ID, PROJECT_NAME, ENTITY)

    # Create the plot
    plot_losses(df)

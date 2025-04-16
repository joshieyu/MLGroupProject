import torch
from torch.utils.data import DataLoader
from model import UNetAutoencoder2D
from dataset import AudioDenoiseDataset
from evaluate import evaluate
from infer import denoise_and_save
import os # Import os
import glob # Import glob

import argparse


def train(model, train_loader, num_epochs=20, device=None, lr=1e-3, checkpoint_interval=5, step_size=10, gamma=0.5):
    criterion = torch.nn.L1Loss()  # Use L1 loss for denoising
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for noisy_spec, clean_spec in train_loader:
            noisy_spec, clean_spec = noisy_spec.to(device), clean_spec.to(device)

            # Zero gradients, perform backward pass, update weights
            optimizer.zero_grad()
            output = model(noisy_spec)
            loss = criterion(output, clean_spec)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Step the scheduler to update the learning rate
        scheduler.step()

        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
    
    return model

def load_model(model_path, device='cpu'):
    # Initialize the model architecture
    model = UNetAutoencoder2D()

    # Load the saved model's state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    return model

def run_model(do_train, device, noisy_dir=None, output_base_dir=None, model_path="denoising_autoencoder.pt"):
    """
    Trains or evaluates the model. For evaluation, uses the specified noisy_dir.

    Args:
        do_train (bool): Whether to train the model.
        device (torch.device): Device to run on.
        noisy_dir (str, optional): Path to the specific noisy dataset directory for evaluation.
                                   Required if do_train is False. Defaults to None.
        output_base_dir (str, optional): Base directory to save evaluation outputs.
                                         Required if do_train is False. Defaults to None.
        model_path (str): Path to save/load the model weights.
    """
    model = None
    clean_dir = "data/clean" # Assuming clean data is always here

    if do_train:
        # Training uses a default noisy directory (or could be made configurable)
        train_noisy_dir = "data/noisy_0_100" # Example: Train on 0.1 noise level
        print(f"Loading training dataset (Clean: {clean_dir}, Noisy: {train_noisy_dir})...")
        if not os.path.exists(train_noisy_dir):
             print(f"ERROR: Training noisy directory not found: {train_noisy_dir}")
             return
        dataset = AudioDenoiseDataset(clean_dir, train_noisy_dir)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        print("Initializing model...")
        model = UNetAutoencoder2D().to(device)

        print("Training model...")
        model = train(model, dataloader, num_epochs=50, device=device) # Adjust epochs as needed

        torch.save(model.state_dict(), model_path)
        print(f"Model saved as '{model_path}'")
    else:
        # Evaluation mode
        if not noisy_dir or not output_base_dir:
            print("ERROR: noisy_dir and output_base_dir must be provided for evaluation.")
            return
        if not os.path.exists(noisy_dir):
             print(f"ERROR: Evaluation noisy directory not found: {noisy_dir}")
             return
        if not os.path.exists(clean_dir):
             print(f"ERROR: Clean directory not found: {clean_dir}")
             return

        print(f"Loading dataset for evaluation (Clean: {clean_dir}, Noisy: {noisy_dir})...")
        dataset = AudioDenoiseDataset(clean_dir, noisy_dir)
        # No DataLoader needed for evaluation as evaluate iterates the dataset directly

        # Load the model (assuming it's already trained)
        if model is None: # Load model only if not already loaded (e.g., in a loop)
             print(f"Loading model from {model_path}...")
             model = load_model(model_path, device)
             model.to(device) # Ensure model is on the correct device

        # --- Evaluation ---
        # Create a specific output directory for this noise level
        noisy_folder_name = os.path.basename(noisy_dir) # e.g., "noisy_0_100"
        eval_save_dir = os.path.join(output_base_dir, f"denoised_{noisy_folder_name}")
        print(f"🔍 Evaluating model on {noisy_folder_name}...")
        print(f"   Saving results to: {eval_save_dir}")
        evaluate(model, dataset, save_dir=eval_save_dir, sample_rate=16000, device=device)
        print("-" * 30)


def getArgs():
    parser = argparse.ArgumentParser(description='Audio denoiser with ML')

    parser.add_argument('-t', '--train', action='store_true', help='Train the model', default=False)
    parser.add_argument('-e', '--eval', action='store_true', help='Evaluate the model on different noise levels', default=False) # Added eval flag
    parser.add_argument('-r', '--run_infer', action='store_true', help='Run inference on a single file', default=False)

    parser.add_argument('-i', '--input_file', type=str, help='Input file for single inference')
    parser.add_argument('-o', '--output_file', type=str, help='Output file for single inference', default='tmp.wav')
    parser.add_argument('--model_path', type=str, default='denoising_autoencoder.pt', help='Path to model weights file')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory containing clean and noisy_* folders')
    parser.add_argument('--results_root', type=str, default='./evaluation_results', help='Base directory to save evaluation results')


    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.train:
        # Run training - uses default noisy dir specified in run_model
        run_model(do_train=True, device=device, model_path=args.model_path)

    elif args.eval:
        # Run evaluation on all found noisy datasets
        print("Starting evaluation process...")
        # Find all noisy directories
        noisy_dirs_pattern = os.path.join(args.data_root, "noisy_*")
        noisy_dirs_found = sorted(glob.glob(noisy_dirs_pattern))

        if not noisy_dirs_found:
            print(f"ERROR: No noisy directories found matching pattern '{noisy_dirs_pattern}'. Cannot evaluate.")
        else:
            print(f"Found noisy directories for evaluation: {noisy_dirs_found}")
            # Load model once before the loop
            print(f"Loading model from {args.model_path} for evaluation...")
            model = load_model(args.model_path, device)
            model.to(device) # Ensure model is on the correct device

            os.makedirs(args.results_root, exist_ok=True) # Ensure base results directory exists

            for noisy_dir in noisy_dirs_found:
                run_model(do_train=False, device=device, noisy_dir=noisy_dir, output_base_dir=args.results_root, model_path=args.model_path) # Pass model if already loaded? No, load_model handles it.

        print("Evaluation finished.")

    elif args.run_infer:
        # Run inference on a single file
        if not args.input_file:
            print("ERROR: --input_file is required for inference mode (-r).")
        else:
            print("Loading model for inference...")
            model = load_model(args.model_path, device)
            model.to(device) # Ensure model is on the correct device
            print(f"Running inference on {args.input_file}...")
            denoise_and_save(model, args.input_file, args.output_file)
            print(f"Denoised output saved to {args.output_file}")
    else:
        print("No action specified. Use -t to train, -e to evaluate, or -r to run inference.")



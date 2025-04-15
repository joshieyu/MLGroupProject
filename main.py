import torch
from torch.utils.data import DataLoader
from model import UNetAutoencoder2D 
from dataset import AudioDenoiseDataset
from evaluate import evaluate 
from infer import denoise_and_save

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

def run_model(do_train, device):
    # Step 1: Set device for training (CPU or GPU)
    model = None
    
    print("Loading dataset...")
    dataset = AudioDenoiseDataset("data/clean", "data/noisy")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    if do_train:
        # Step 3: Initialize model and move to device
        print("Initializing model...")
        model = UNetAutoencoder2D().to(device)

        # Step 4: Train the model
        print("Training model...")
        model = train(model, dataloader, num_epochs=50, device=device)

        # Step 5: Save model
        torch.save(model.state_dict(), "denoising_autoencoder.pt")
        print("Model saved as 'denoising_autoencoder.pt'")
    else:
        model = load_model("denoising_autoencoder.pt", device)

    # Step 6: Evaluate the model
    print("🔍 Evaluating model...")
    evaluate(model, dataset, save_dir="denoised_outputs", sample_rate=16000, device=device)

def getArgs():
    parser = argparse.ArgumentParser(description='Audio denoiser with ML')

    parser.add_argument('-t', '--train', action='store_true', help='To train or not to train', default=False)
    parser.add_argument('-r', '--run_infer', action='store_true', help='Try the model on a new file', default=False)

    parser.add_argument('-i', '--input_file', type=str, help='Input file to run model on')
    parser.add_argument('-o', '--output_file', type=str, help='Location of output file', default='tmp.wav')

    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.run_infer:
        run_model(args.train, device)
    else:
        model = load_model("denoising_autoencoder.pt", device)
        denoise_and_save(model, args.input_file, args.output_file)



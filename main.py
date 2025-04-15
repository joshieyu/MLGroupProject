import torch
from torch.utils.data import DataLoader
from model import UNetAutoencoder2D 
from dataset import AudioDenoiseDataset
from evaluate import evaluate 

def train(model, train_loader, num_epochs=20, device=None, lr=1e-3):
    criterion = torch.nn.L1Loss()  # Use L1 loss for denoising
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for noisy_spec, clean_spec in train_loader:
            # Move to device if using GPU
            noisy_spec, clean_spec = noisy_spec.to(device), clean_spec.to(device)

            # Zero gradients, perform backward pass, update weights
            optimizer.zero_grad()
            output = model(noisy_spec)
            loss = criterion(output, clean_spec)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
    
    return model

def load_model(model_path, device='cpu'):
    # Initialize the model architecture
    model = UNetAutoencoder2D()

    # Load the saved model's state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set the model to evaluation mode
    model.eval()

    return model

def main(trained):
    # Step 1: Set device for training (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    
    print("Loading dataset...")
    dataset = AudioDenoiseDataset("data/clean", "data/noisy")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    if not trained:

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

if __name__ == "__main__":
    main(True)


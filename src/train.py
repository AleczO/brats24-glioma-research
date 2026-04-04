import torch
import os
from tqdm import tqdm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from src import get_brats_loaders, get_model

# AMD GPU Override for RX 6800M (RDNA 2)
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

def run_training():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "./data_inventory.csv"
    best_model_path = "best_model.pth"
    latest_model_path = "latest_model.pth"
    
    epochs = 50
    val_interval = 1
    batch_size = 2
    learning_rate = 1e-4
    
    # 2. Component Initialization
    print(f"Initializing training on: {device}")
    train_loader, val_loader = get_brats_loaders(csv_path, batch_size=batch_size)
    model = get_model().to(device)
    
    # --- Resume prevous model ---
    if os.path.exists(best_model_path):
        print(f"Znaleziono zapisany model: {best_model_path}. Wczytywanie wag...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Model wczytany pomyślnie. Kontynuujemy od poziomu ~0.58 Dice!")
    else:
        print("Nie znaleziono checkpointu. Rozpoczynam naukę od zera.")
    # ----------------------------------------------

    loss_function = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    scaler = torch.amp.GradScaler('cuda')

    # Jeśli wczytałeś model, wypadałoby zacząć od 0.5819, żeby nie nadpisać go gorszym wynikiem
    best_dice = 0.5819 if os.path.exists(best_model_path) else 0
    
    # 3. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch_data in progress_bar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{epoch_loss/step:.4f}"})

        torch.save(model.state_dict(), latest_model_path)
        
        # 4. Validation using Sliding Window
        if (epoch + 1) % val_interval == 0:
            model.eval()
            dice_metric.reset()
            
            val_progress = tqdm(val_loader, desc="Validation", leave=False)
            
            with torch.no_grad():
                for val_data in val_progress:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    with torch.amp.autocast('cuda'):
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs, 
                            roi_size=(128, 128, 128), 
                            sw_batch_size=1, 
                            predictor=model,
                            overlap=0.5
                        )
                    
                    val_outputs = [torch.sigmoid(i) > 0.5 for i in val_outputs]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                current_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f"\n--- Epoka {epoch+1}: Dice = {current_dice:.4f} ---")
                
                if current_dice > best_dice:
                    best_dice = current_dice
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Nowy rekord! Zapisano best_model.pth (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    run_training()
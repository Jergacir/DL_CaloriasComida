import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def entrenar_epoca(modelo, dataloader, criterion, optimizer, device):
    modelo.train()
    perdida_total = 0
    
    for batch in tqdm(dataloader, desc="Entrenamiento"):
        imagenes = batch['imagen'].to(device)
        targets = torch.stack([batch['masa'], batch['calorias']], dim=1).to(device)
        
        # Forward
        outputs = modelo(imagenes)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        perdida_total += loss.item()
    
    return perdida_total / len(dataloader)

def validar_epoca(modelo, dataloader, criterion, device):
    modelo.eval()
    perdida_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            imagenes = batch['imagen'].to(device)
            targets = torch.stack([batch['masa'], batch['calorias']], dim=1).to(device)
            
            outputs = modelo(imagenes)
            loss = criterion(outputs, targets)
            
            perdida_total += loss.item()
    
    return perdida_total / len(dataloader)

def entrenar_modelo(modelo, train_loader, val_loader, config):
    device = config.DEVICE
    modelo = modelo.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=config.LEARNING_RATE, 
                          weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    mejor_val_loss = float('inf')
    historial = {'train_loss': [], 'val_loss': []}
    
    print(f"Entrenando en: {device}")
    print(f"Épocas: {config.EPOCAS}\n")
    
    for epoca in range(config.EPOCAS):
        print(f"\n{'='*50}")
        print(f"Época {epoca+1}/{config.EPOCAS}")
        print(f"{'='*50}")
        
        # Entrenar
        train_loss = entrenar_epoca(modelo, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss = validar_epoca(modelo, val_loader, criterion, device)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Guardar historial
        historial['train_loss'].append(train_loss)
        historial['val_loss'].append(val_loss)
        
        # Guardar mejor modelo
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            torch.save({
                'epoca': epoca,
                'model_state_dict': modelo.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, 'mejor_modelo.pth'))
            print(f"✓ Mejor modelo guardado (val_loss: {val_loss:.4f})")
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return historial

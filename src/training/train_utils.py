import torch 
import os 
import json 
import numpy as np 

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0 
    correct = 0 

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def validate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def save_history (history, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(save_path, "w") as f:
        json.dump(history, f)
    
    print(f'History saved to {save_path}')

def train(
        model,
        model_name,
        experiment,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer, 
        scheduler,
        num_epochs,
        patience,
        device,
        save_dir,
        logger = None 
):
    '''
    Train the model for a specified number of epochs with early stopping.
    Saves the best model and the last model in the specified directory.
    '''
    os.makedirs(save_dir, exist_ok=True)

    best_loss = np.inf
    patience_counter = 0 

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )

        val_loss, val_accuracy = validate(
            model, valid_loader, loss_fn, device
        )

        if scheduler:
            scheduler.step(val_loss)
        
        msg = (
            f'Epoch [{epoch}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
        )
        if logger:
            logger.info(msg)
        print(msg)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_accuracy
        })

        idx_to_class = {v: k for k, v in train_loader.dataset.class_to_idx.items()}

        # Save best model 
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0 
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'idx_to_class': idx_to_class,
                'class_to_idx': train_loader.dataset.class_to_idx,
                'num_classes': len(train_loader.dataset.classes),
                'experiment': experiment
            }, best_model_path)
            msg = (f'Best model saved to {best_model_path}')
            if logger:
                logger.info(msg)
            print(msg)
        else:
            patience_counter += 1 
        
        if patience_counter >= patience :
            msg = 'Early stopping triggered'
            if logger:
                logger.info(msg)
            print(msg)
            break
    
    # Save History
    save_history(history, save_dir, model_name)

    return model, history 
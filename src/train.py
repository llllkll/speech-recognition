import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from src.data import LibriSpeechDataset, TextTransform, collate_fn
from src.models import SpeechRecognitionModel, greedy_decode
from src.utils import WanDBLogger, cer, wer, global_pruning, inference_speed, train_distill

SEED = 7
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@hydra.main(config_path="../configs", config_name="config")
def main(config):
    logger = WanDBLogger(dict(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_dataloaders(config)
    if config.distillation.enable:
        teacher = SpeechRecognitionModel(**config["model"]).to(device)
        teacher.load_state_dict(torch.load(config.distillation.teacher_checkpoint, map_location=device))
        student = SpeechRecognitionModel(**config.distillation.student_arch).to(device)
        model = student
        print(f"Teacher model loaded. Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
    else:
        model = SpeechRecognitionModel(**config["model"]).to(device)
        print(f"Standard training. Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    model = SpeechRecognitionModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    n_epochs = config["train"]["epochs"]
    if config.pruning.enable:
        n_epochs += config.pruning.fine_tune_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["train"]["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=n_epochs,
        anneal_strategy='linear')
    criterion = torch.nn.CTCLoss(blank=28).to(device)
    best_wer = float('inf')
    for epoch in range(config["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}")
        if config.distillation.enable:
            train_loss = train_distill(
                model, teacher, device, train_loader, 
                criterion, optimizer, scheduler, logger
                )
        else:
            train_loss = train_epoch(
                model, device, train_loader, 
                criterion, optimizer, scheduler, logger
            )
        
        val_loss, val_cer, val_wer = validate_epoch(
            model, device, val_loader, criterion, logger
        )
        
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), f"{config['train']['save_dir']}/best_model.pth")
            logger.log_checkpoint(f"{config['train']['save_dir']}/best_model.pth")
            
    if config.pruning.enable:
        global_pruning(model, config.pruning.rate)
        
        print(f"\nFine-tuning after pruning ({config.pruning.rate*100}% sparsity)")
        val_loss, val_cer, val_wer = 0, 0, 0
        for epoch in range(config.pruning.fine_tune_epochs):
            train_loss = train_epoch(
                model, device, train_loader, 
                criterion, optimizer, scheduler, logger)
        
            val_loss, val_cer, val_wer = validate_epoch(model, device, val_loader, criterion, logger)

        time_inf = inference_speed(model=model, test_loader=val_loader, dtype="None", device="cpu")
        logger.log_metrics({
            "inference_time": time_inf
        })
        torch.save(model.state_dict(), f"{config['train']['save_dir']}/pruned_model.pth")
        logger.log_checkpoint(f"{config['train']['save_dir']}/pruned_model.pth")
    time_inf = inference_speed(model=model, test_loader=val_loader, dtype="None", device="cpu")
    logger.log_metrics({
        "inference_time": time_inf
    })    
            


def get_dataloaders(config):
    text_transform = TextTransform()
    
    train_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                      for url in config["data"]["urls"]["train"]]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, text_transform, "train")
    )
   
    val_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                    for url in config["data"]["urls"]["dev"]]
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, text_transform, "valid")
    )
    
    return train_loader, val_loader

def train_epoch(model, device, loader, criterion, optimizer, scheduler, logger):
    model.train()
    total_loss = 0.0
    data_len = len(loader.dataset)
    
    for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(loader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        
        loss = criterion(output, labels, input_lengths, label_lengths)
        total_loss += loss.item()
        loss.backward()
        logger.log_metrics({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0]
            })
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(loader), loss.item()))
    return total_loss / len(loader)

def validate_epoch(model, device, loader, criterion, logger):
    model.eval()
    val_loss = 0.0
    val_cer, val_wer = [], []
    examples_table = []
    
    with torch.no_grad():
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(loader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            output = model(spectrograms)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)
            
            loss = criterion(output, labels, input_lengths, label_lengths)
            val_loss += loss.item() / len(loader)
            
            decoded_preds, decoded_targets = greedy_decode(
                output.transpose(0, 1), 
                labels, 
                label_lengths, 
                TextTransform()
            )
            
            if batch_idx == 0:
                print(f"target: {decoded_targets[0]}")
                print(f"Predict: {decoded_preds[0]}")
                for i in range(min(5, len(decoded_preds))):
                    examples_table.append([
                        decoded_targets[i],
                        decoded_preds[i]
                    ])
            for j in range(len(decoded_preds)):
                val_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                val_wer.append(wer(decoded_targets[j], decoded_preds[j]))
    

    avg_cer = sum(val_cer) / len(val_cer)
    avg_wer = sum(val_wer) / len(val_wer)
    table = wandb.Table(
        columns=["Target Text", "Predicted Text"],
        data=examples_table
    )
    logger.log_metrics({
        "test/loss": val_loss,
        "test/cer": avg_cer,
        "test/wer": avg_wer,
        "test/examples": table
    })
    
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(val_loss, avg_cer, avg_wer))

    
    return val_loss, avg_cer, avg_wer

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="configs/config.yaml")
    # args = parser.parse_args()
    main()import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from src.data import LibriSpeechDataset, TextTransform, collate_fn
from src.models import SpeechRecognitionModel, greedy_decode
from src.utils import WanDBLogger, cer, wer, global_pruning, inference_speed, train_distill

SEED = 7
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@hydra.main(config_path="../configs", config_name="config")
def main(config):
    logger = WanDBLogger(dict(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_dataloaders(config)
    if config.distillation.enable:
        teacher = SpeechRecognitionModel(**config["model"]).to(device)
        teacher.load_state_dict(torch.load(config.distillation.teacher_checkpoint, map_location=device))
        student = SpeechRecognitionModel(**config.distillation.student_arch).to(device)
        model = student
        print(f"Teacher model loaded. Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
    else:
        model = SpeechRecognitionModel(**config["model"]).to(device)
        print(f"Standard training. Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    model = SpeechRecognitionModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    n_epochs = config["train"]["epochs"]
    if config.pruning.enable:
        n_epochs += config.pruning.fine_tune_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["train"]["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=n_epochs,
        anneal_strategy='linear')
    criterion = torch.nn.CTCLoss(blank=28).to(device)
    best_wer = float('inf')
    for epoch in range(config["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}")
        if config.distillation.enable:
            train_loss = train_distill(
                model, teacher, device, train_loader, 
                criterion, optimizer, scheduler, logger
                )
        else:
            train_loss = train_epoch(
                model, device, train_loader, 
                criterion, optimizer, scheduler, logger
            )
        
        val_loss, val_cer, val_wer = validate_epoch(
            model, device, val_loader, criterion, logger
        )
        
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), f"{config['train']['save_dir']}/best_model.pth")
            logger.log_checkpoint(f"{config['train']['save_dir']}/best_model.pth")
            
    if config.pruning.enable:
        global_pruning(model, config.pruning.rate)
        
        print(f"\nFine-tuning after pruning ({config.pruning.rate*100}% sparsity)")
        val_loss, val_cer, val_wer = 0, 0, 0
        for epoch in range(config.pruning.fine_tune_epochs):
            train_loss = train_epoch(
                model, device, train_loader, 
                criterion, optimizer, scheduler, logger)
        
            val_loss, val_cer, val_wer = validate_epoch(model, device, val_loader, criterion, logger)

        time_inf = inference_speed(model=model, test_loader=val_loader, dtype="None", device="cpu")
        logger.log_metrics({
            "inference_time": time_inf
        })
        torch.save(model.state_dict(), f"{config['train']['save_dir']}/pruned_model.pth")
        logger.log_checkpoint(f"{config['train']['save_dir']}/pruned_model.pth")
    time_inf = inference_speed(model=model, test_loader=val_loader, dtype="None", device="cpu")
    logger.log_metrics({
        "inference_time": time_inf
    })    
            


def get_dataloaders(config):
    text_transform = TextTransform()
    
    train_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                      for url in config["data"]["urls"]["train"]]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, text_transform, "train")
    )
   
    val_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                    for url in config["data"]["urls"]["dev"]]
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, text_transform, "valid")
    )
    
    return train_loader, val_loader

def train_epoch(model, device, loader, criterion, optimizer, scheduler, logger):
    model.train()
    total_loss = 0.0
    data_len = len(loader.dataset)
    
    for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(loader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)
        
        loss = criterion(output, labels, input_lengths, label_lengths)
        total_loss += loss.item()
        loss.backward()
        logger.log_metrics({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0]
            })
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(loader), loss.item()))
    return total_loss / len(loader)

def validate_epoch(model, device, loader, criterion, logger):
    model.eval()
    val_loss = 0.0
    val_cer, val_wer = [], []
    examples_table = []
    
    with torch.no_grad():
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(loader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            output = model(spectrograms)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)
            
            loss = criterion(output, labels, input_lengths, label_lengths)
            val_loss += loss.item() / len(loader)
            
            decoded_preds, decoded_targets = greedy_decode(
                output.transpose(0, 1), 
                labels, 
                label_lengths, 
                TextTransform()
            )
            
            if batch_idx == 0:
                print(f"target: {decoded_targets[0]}")
                print(f"Predict: {decoded_preds[0]}")
                for i in range(min(5, len(decoded_preds))):
                    examples_table.append([
                        decoded_targets[i],
                        decoded_preds[i]
                    ])
            for j in range(len(decoded_preds)):
                val_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                val_wer.append(wer(decoded_targets[j], decoded_preds[j]))
    

    avg_cer = sum(val_cer) / len(val_cer)
    avg_wer = sum(val_wer) / len(val_wer)
    table = wandb.Table(
        columns=["Target Text", "Predicted Text"],
        data=examples_table
    )
    logger.log_metrics({
        "test/loss": val_loss,
        "test/cer": avg_cer,
        "test/wer": avg_wer,
        "test/examples": table
    })
    
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(val_loss, avg_cer, avg_wer))

    
    return val_loss, avg_cer, avg_wer

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="configs/config.yaml")
    # args = parser.parse_args()
    main()

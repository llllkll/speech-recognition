import torch
import torch.nn.functional as F

def train_distill(model,teacher_model,device, train_loader, criterion, optimizer, scheduler, logger, temperature=3.0, alpha=0.7,epoch=None):
    model.train()
    teacher_model.eval()
    
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_distill_loss = 0.0
    data_len = len(train_loader.dataset)
    
    for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher_model(spectrograms)
        
        student_logits = model(spectrograms)

        log_probs = F.log_softmax(student_logits, dim=2)
        output = log_probs.transpose(0, 1)  # (time, batch, n_class)
        ctc_loss = criterion(
            output,
            labels,
            input_lengths,
            label_lengths
        )

        soft_labels = F.softmax(teacher_logits / temperature, dim=2)
        soft_predictions = F.log_softmax(student_logits / temperature, dim=2)
        distill_loss = F.kl_div(
            soft_predictions,
            soft_labels,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        loss = alpha * ctc_loss + (1 - alpha) * distill_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_ctc_loss += ctc_loss.item()
        total_distill_loss += distill_loss.item()
        
        if batch_idx % 100 == 0 or batch_idx == data_len:
            logger.log_metrics({
                "train/loss": ctc_loss.item(),
                "train/lr": scheduler.get_last_lr()[0]
            })
            
            print(f'Epoch: {epoch} [{batch_idx * len(spectrograms)}/{data_len} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] '
                  f'Loss: {loss.item():.6f} '
                  f'CTC: {ctc_loss.item():.6f} '
                  f'Distill: {distill_loss.item():.6f}')
    
    return {
        "total_loss": total_loss / len(train_loader),
        "ctc_loss": total_ctc_loss / len(train_loader),
        "distill_loss": total_distill_loss / len(train_loader)
    }

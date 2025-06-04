import hydra
import torch
import wandb
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.data import LibriSpeechDataset, TextTransform, collate_fn
from src.models import SpeechRecognitionModel, greedy_decode
from src.utils import WanDBLogger, cer, wer, quantize_model, inference_speed


@hydra.main(config_path="../configs", config_name="config")
def main(config):
    checkpoint = config["test"]["checkpoint"]
    logger = WanDBLogger(dict(config))
    logger.watch_model = False
    text_transform = TextTransform()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechRecognitionModel(**config["model"]).to("cpu")
    try:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    except Exception as e:
        msg = f"Error loading checkpoint: {e!s}"
        raise RuntimeError(msg)

    test_datasets = [LibriSpeechDataset(config["data"]["data_dir"], url)
                    for url in config["data"]["urls"]["test"]]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, text_transform, "test")
    )
    
    criterion = torch.nn.CTCLoss(blank=28).to(device)
    
    if config.quantization.enable:
        example_input = next(iter(test_loader))[0][0].unsqueeze(0).to("cpu")
        
        quantized_model = quantize_model(
            model=model,
            example_input=example_input,
            dtype=config.quantization.dtype
        )
        
        torch.save(quantized_model.state_dict(), f"{config['train']['save_dir']}/quantized_model.pth")
        logger.log_checkpoint(f"{config['train']['save_dir']}/quantized_model.pth")
        final = SpeechRecognitionModel(**config["model"])
        final.load_state_dict(torch.load(config["quantization"]["checkpoint"]))
        time = inference_speed(model=quantized_model, test_loader=test_loader, dtype=config.quantization.dtype, device=device)
        
        logger.log_metrics({
            "inference_time": time
        })
        test(quantized_model, device, test_loader, criterion, logger)
    else:
        time = inference_speed(model=model, test_loader=test_loader, dtype="None", device=device)
        logger.log_metrics({
            "inference_time": time
        })
        test(model, device, test_loader, criterion, logger)
        
        



def test(model, device, test_loader, criterion, logger) -> None:

    text_transform = TextTransform()
    test_loss = 0.0
    test_cer, test_wer = [], []
    total_samples = 0
    examples = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(test_loader):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            output = model(spectrograms)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)
            
            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)
            
            decoded_preds, decoded_targets = greedy_decode(
                output.transpose(0, 1), 
                labels, 
                label_lengths, 
                text_transform
            )

            if batch_idx == 0:
                print(f"target: {decoded_targets[0]}")
                print(f"Predict: {decoded_preds[0]}")
                for i in range(min(5, len(decoded_preds))):
                    examples.append([
                        decoded_targets[i],
                        decoded_preds[i]
                    ])
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))
                
    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    table = wandb.Table(
        columns=["Target Text", "Predicted Text"],
        data=examples
    )
    logger.log_metrics({
        "test/loss": test_loss,
        "test/cer": avg_cer,
        "test/wer": avg_wer,
        "test/examples": table
    })
    
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))            


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="ASR Model Testing")
    # parser.add_argument("--config", default="configs/config.yaml")
    # parser.add_argument("--checkpoint", required=True)
    # args = parser.parse_args()
    main()

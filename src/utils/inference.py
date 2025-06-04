import time
import torch
from torch.quantization import quantize_dynamic
import torch.nn.utils.prune as prune

torch.backends.quantized.engine = 'fbgemm'

def quantize_model(model, example_input, dtype='fp16', backend='fbgemm'):
    quantized_model = model
    
    if dtype == 'int8':
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model.qconfig = qconfig
        
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8,
            inplace=False
        )
        example_input= example_input.to("cpu")
    elif dtype == 'fp16':
        if torch.cuda.is_available():
            quantized_model = model.half().cpu()
            example_input = example_input.half().cpu()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    quantized_model.eval()
    with torch.no_grad():
        quantized_model(example_input)
    
    return quantized_model

def global_pruning(model, pruning_rate=0.2, method="l1_unstructured"):
    parameters_to_prune = [
        (module, 'weight') 
        for module in model.modules() 
        if isinstance(module, torch.nn.Conv2d)
    ]
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate
    )

    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    return model
    


def inference_speed(model, test_loader, dtype, device, num_examples=5):
    model.to(device)
    model.eval()
    times = []
    example_batch = next(iter(test_loader))
    spectrograms, _, _, _ = example_batch
    if dtype == 'fp16':
        inputs = spectrograms[:num_examples].half().cpu()
    else:    
        inputs = spectrograms[:num_examples].to(device)

    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(100):
            _ = model(inputs)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    total_time = (end_time - start_time) * 1000
    time_per_batch = total_time / 100
    time_per_sample = time_per_batch / num_examples
    
    # Результаты
    print(f"Inference speed:")
    print(f"- Total runs: 100 batches")
    print(f"- Batch size: {num_examples}")
    print(f"- Time per batch: {time_per_batch:.2f} ms")
    print(f"- Time per sample: {time_per_sample:.2f} ms")
    
    return time_per_sample

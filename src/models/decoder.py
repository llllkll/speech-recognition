import torch
import wandb


def greedy_decode(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []

    example_pred = []
    example_target = []

    for i, args in enumerate(arg_maxes):
        decode = []
        target_indices = labels[i][:label_lengths[i]].tolist()
        target_text = text_transform.int_to_text(target_indices)
        targets.append(target_text)
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decoded_text = text_transform.int_to_text(decode)
        decodes.append(decoded_text)
        if i == 0:
            example_pred = decode
            example_target = target_indices

    return decodes, targets

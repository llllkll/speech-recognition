import os

import torchaudio
from torch.utils.data import Dataset


class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir, url, download=True) -> None:
        folder = os.path.join(data_dir, "LibriSpeech", url)
        dl = download and not os.path.isdir(folder)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=data_dir,
            url=url,
            download=dl,
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

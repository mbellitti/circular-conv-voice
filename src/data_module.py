import lightning as L
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split
from pprint import pprint
import torch.nn as nn


class VoiceDataset(Dataset):

    def __init__(self, config):

        self.config = config

        # make a dataframe with uid and path to audiofile
        self.audio_paths = pd.DataFrame(
            [
                {"uid": audio_path.stem, "audio_path": audio_path}
                for audio_path in Path(self.config.data.audio_dir).glob(
                    f"*.{config.data.audio_extension}"
                )
            ]
        )

        # labels, demographics, etc.
        self.tabular_data = pd.read_csv(self.config.data.tabular_data)

        self.tabular_data = self.tabular_data[self.tabular_data['filesize_kb'] > 150]

        self.tabular_data['label'] = self.tabular_data['label'].astype(float)

        # merge into a single dataframe with uid, metadata, and path to audios
        self.data = pd.merge(self.audio_paths, self.tabular_data, on="uid")

        print(
            f"Found {len(self.data)} audio files with valid labels. Here is a sample:"
        )

        pprint(self[0])
        print()

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        datapoint_dict = {}

        datapoint_dict["age"] = self.data.iloc[idx]["age"]
        datapoint_dict["label"] = self.data.iloc[idx]["label"]

        # Load audio. By default the shape is [channel,time]
        # it might be worth for wav type to work with integers
        # Even if there's a single channel it's better not to squeeze it out, the convolutional layers
        # assume it's there.
        waveform, sampling_rate = torchaudio.load(self.data.iloc[idx]["audio_path"])

        assert sampling_rate == 48000, "Found non 48kHz sampling rate."

        seq_length = waveform.shape[-1]

        # circular pad all sequences to be the same length
        padder = nn.CircularPad1d((0, self.config.data.pad_to_length - seq_length))
        waveform = padder(waveform)

        datapoint_dict["waveform"] = waveform
        datapoint_dict["seq_length"] = seq_length
        datapoint_dict["sampling_rate"] = sampling_rate

        return datapoint_dict


class VoiceDataModule(L.LightningDataModule):
    def __init__(self, config):

        super().__init__()
        self.config = config

    def setup(self, stage: str):

        self.dataset = VoiceDataset(self.config)

        # split into training and validation sets
        train_idx, val_idx = train_test_split(
            list(range(len(self.dataset))),
            test_size=self.config.data.val_size,
            random_state=self.config.data.train_test_split_seed,
            shuffle=True,
            stratify=[
                self.dataset.data.iloc[i]["label"] for i in range(len(self.dataset))
            ],
        )

        # print summaries of train and test set

        print("Training set:")
        print("Total samples: ", len(train_idx))
        print("Class distribution: ")
        print(
            self.dataset.data.iloc[train_idx]["label"]
            .value_counts(normalize=True)
            .to_frame()
            .sort_index(),
            end="\n\n",
        )

        print("Validation set:")
        print("Total samples: ", len(val_idx))
        print("Class distribution: ")
        print(
            self.dataset.data.iloc[val_idx]["label"]
            .value_counts(normalize=True)
            .to_frame()
            .sort_index(),
            end="\n\n",
        )

        self.train_dataset = Subset(self.dataset, train_idx)

        self.val_dataset = Subset(self.dataset, val_idx)

        # print summary of train and test datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            pin_memory=self.config.data.pin_memory,
            shuffle=True,
            num_workers=self.config.data.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            pin_memory=self.config.data.pin_memory,
            shuffle=False,
            num_workers=self.config.data.dataloader_num_workers,
        )

    # add test_dataloader, test_dataloader, predict_dataloader if needed

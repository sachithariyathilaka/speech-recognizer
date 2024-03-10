from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader

from module.dataset import collate_fn_padd, Data


class SpeechModule(LightningModule):

    def __init__(self, model, train_json_path, valid_json_path):
        super(SpeechModule, self).__init__()
        self.scheduler = None
        self.optimizer = None
        self.model = model
        self.output = None
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.train_json_path = train_json_path
        self.valid_json_path = valid_json_path

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), 1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.50,
            patience=6)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }

    def step(self, batch):
        spectrogram, labels, input_lengths, label_lengths = batch
        bs = spectrogram.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrogram, (hn, c0))
        output = functional.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update({})
        train_dataset = Data(json_path=self.train_json_path, **d_params)
        return DataLoader(dataset=train_dataset,
                          batch_size=64,
                          num_workers=1,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        self.output = self.step(batch)
        return {'val_loss': self.output}

    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update({})
        test_dataset = Data(json_path=self.valid_json_path, **d_params, valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=64,
                          num_workers=0,
                          collate_fn=collate_fn_padd,
                          pin_memory=True)

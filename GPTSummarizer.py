from typing import Optional, Callable

import pytorch_lightning as pl
from torch import nn
import torch
import torch.utils.data
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2LMHeadModel, AdamW
import torch.nn.functional as F
class GPTSummarizerPL(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Warning! Labels are shifted inside the model
        self.gpt = GPT2LMHeadModel.from_pretrained(hparams['pretrained_model_path'])
        self.hparams = hparams
        self.save_hyperparameters(hparams)

    def forward(self, encoded_ids, labels=None, segments=None):
        batch_size=encoded_ids.size(0)
        if labels is not None:
            logits = self.gpt.forward(encoded_ids[:,:-1])[0]

            targets = labels[:,1:]
            segments = segments[:,1:].reshape(-1)
            losses = F.cross_entropy(logits.reshape(batch_size * logits.size(1), -1),
                                   targets.reshape(-1), reduction='none')

            w1 = self.hparams['content_loss_weight']
            w2 = self.hparams['summary_loss_weight']

            coefs = (w1 * (segments == 0) + w2 * (segments == 1)).type(torch.float32)

            total = len(coefs.view(-1))
            loss = coefs.dot(losses) / total
            return loss, logits

        else:
            logits = self.gpt.forward(encoded_ids)[0]
            return logits

    def training_step(self, batch, batch_idx):
        """target ids are shifted inside the model.
        In most cases input_ids and target_ids are equal.
        Use index -100 to ignore loss for some labels
        """
        input_ids, target_ids, segment_ids = batch

        loss, logits = self.forward(input_ids, target_ids, segment_ids)
        self.log('training_step_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, target_ids, segment_ids = batch
        loss, logits = self.forward(input_ids, target_ids, segment_ids)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss.item()}

    def validation_epoch_end(self, outputs: list):
        mean_val_loss = 0
        for output in outputs:
            mean_val_loss += output['val_loss']
        mean_val_loss /= len(outputs)
        self.log('avg_val_loss', mean_val_loss, prog_bar=True)

    def configure_optimizers(self):
        def warmup_lambda(step):
            lr_scale = min(1., float(step + 1.0) / self.hparams['warmup_steps'])
            return lr_scale

        optimizer = AdamW(self.parameters(), lr=self.hparams['learning_rate'])
        scheduler = LambdaLR(optimizer, warmup_lambda)
        return [optimizer],[{'scheduler': scheduler, 'interval': 'step'}]

    # def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer: Optimizer = None,
    #                    optimizer_idx: int = None, optimizer_closure: Optional[Callable] = None, on_tpu: bool = None,
    #                    using_native_amp: bool = None, using_lbfgs: bool = None) -> None:
    #     current_step = self.trainer.global_step
    #     warmup_steps = self.hparams['warmup_steps'] if self.hparams['warmup_steps'] else 0
    #     warmup_stage = current_step < warmup_steps
    #
    #     if warmup_stage:
    #         lr_scale = min(1., float(current_step + 1.0) / warmup_steps)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams['learning_rate']
    #
    #     elif self.hparams['linear_decay_steps']:
    #         current_decay_step = current_step - warmup_steps
    #         lr_scale = max(0., 1 - float(current_decay_step) / self.hparams['linear_decay_steps'])
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams['learning_rate']
    #
    #     super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,
    #                            using_lbfgs)




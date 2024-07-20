from typing import Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
from audiotools import AudioSignal
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig

from sq_codec.losses.disc_loss import MSEDiscriminatorLoss
from sq_codec.losses.gen_loss import GeneratorSTFTLoss
from sq_codec.models.sq_codec import SQCodec
from sq_codec.modules.discriminator import MultiFrequencyDiscriminator


class SQCodecLightningModule(L.LightningModule):
    def __init__(
        self,
        generator: DictConfig,
        discriminator: DictConfig,
        gen_loss: DictConfig,
        disc_loss: DictConfig,
        opt_g: DictConfig,
        opt_d: DictConfig,
        schd_g: Optional[DictConfig],
        schd_d: Optional[DictConfig],
        warmup_steps: int = 0,
        num_samples: int = 4,
        sample_freq: int = 5000,
        do_compile: bool = False,
    ):
        super().__init__()

        self.generator: SQCodec = hydra.utils.instantiate(generator)
        self.discriminator: MultiFrequencyDiscriminator = hydra.utils.instantiate(discriminator)

        self.gen_loss: GeneratorSTFTLoss = hydra.utils.instantiate(gen_loss)
        self.disc_loss: MSEDiscriminatorLoss = hydra.utils.instantiate(disc_loss)

        self.opt_g = opt_g
        self.opt_d = opt_d
        self.schd_g = schd_g
        self.schd_d = schd_d

        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.sample_freq = sample_freq

        self.automatic_optimization = False
        self.first_log = True
        if do_compile:
            self.generator = torch.compile(self.generator, mode="max-autotune")

        self.encoder_grad_mult = self.generator.encoder_grad_mult

    def configure_optimizers(self):
        opt_g = hydra.utils.instantiate(self.opt_g, params=self.generator.parameters())
        opt_d = hydra.utils.instantiate(self.opt_d, params=self.discriminator.parameters())

        if self.schd_g is None and self.schd_d is None:
            return [opt_g, opt_d]

        schd_g = hydra.utils.instantiate(self.schd_g, opt_g)
        schd_d = hydra.utils.instantiate(self.schd_d, opt_d)
        return [opt_g, opt_d], [schd_g, schd_d]

    def on_train_epoch_end(self):
        if self.lr_schedulers() is not None:
            schd_g, schd_d = self.lr_schedulers()
            schd_g.step()
            schd_d.step()

    def training_step(self, batch, batch_idx):
        if self.global_step % self.sample_freq == 0:
            self.log_audio()
        audio = batch

        opt_g, opt_d = self.optimizers()
        opt_d._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_d._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        grad_mult_decay = 1 - self.global_step / (self.warmup_steps + 1)
        self.generator.encoder_grad_mult = max(1.0, self.encoder_grad_mult * grad_mult_decay)

        pred = self.generator(audio)

        output_real, output_fake, fmap_real, fmap_fake = {}, {}, {}, {}
        d_name = "mfd"
        output_real[d_name], output_fake[d_name], fmap_real[d_name], fmap_fake[d_name] = self.discriminator(audio, pred)

        use_adv_loss = self.global_step >= self.warmup_steps
        g_loss, g_loss_item = self.gen_loss(
            audio, pred, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss=use_adv_loss
        )
        g_loss_item["Train/g_loss"] = g_loss

        self.manual_backward(g_loss)
        gen_grad_norm = grad_norm(self.generator, norm_type=2)["grad_2.0_norm_total"]
        self.log("other/gen_grad_norm", gen_grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        enc_grad_norm = grad_norm(self.generator.encoder, norm_type=2)["grad_2.0_norm_total"]
        self.log("other/enc_grad_norm", enc_grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        dec_grad_norm = grad_norm(self.generator.decoder, norm_type=2)["grad_2.0_norm_total"]
        self.log("other/dec_grad_norm", dec_grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.clip_gradients(opt_g, gradient_clip_val=100.0, gradient_clip_algorithm="norm")
        opt_g.step()
        opt_g.zero_grad()
        self.discriminator.zero_grad()

        for k, v in g_loss_item.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if not use_adv_loss:
            return g_loss

        pred = self.generator(audio)
        output_real, output_fake, _, _ = self.discriminator(audio, pred.detach())
        d_loss = self.disc_loss(output_real, output_fake)
        self.manual_backward(d_loss)
        disc_grad_norm = grad_norm(self.discriminator, norm_type=2)["grad_2.0_norm_total"]
        self.log("other/disc_grad_norm", disc_grad_norm, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.clip_gradients(opt_d, gradient_clip_val=100.0, gradient_clip_algorithm="norm")
        opt_d.step()
        opt_d.zero_grad()

        self.log("Train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        audio = batch
        pred = self.generator(audio)

        output_real, output_fake, fmap_real, fmap_fake = {}, {}, {}, {}
        d_name = "mfd"
        output_real[d_name], output_fake[d_name], fmap_real[d_name], fmap_fake[d_name] = self.discriminator(audio, pred)

        use_adv_loss = self.global_step >= self.warmup_steps
        g_loss, loss_item = self.gen_loss(
            audio, pred, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss=use_adv_loss
        )
        loss_item["Train/g_loss"] = g_loss

        output_real, output_fake, _, _ = self.discriminator(audio, pred.detach())
        d_loss = self.disc_loss(output_real, output_fake)
        loss_item["Train/d_loss"] = d_loss

        for k, v in loss_item.items():
            self.log(k.replace("Train", "Valid"), v, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    @torch.no_grad()
    def log_audio(self):
        torch.cuda.empty_cache()
        if self.global_rank != 0:
            return

        self.generator.eval()
        writer = self.logger.experiment
        for i in range(self.num_samples):
            audio = self.trainer.datamodule.val_ds.get_full_audio(i)
            if self.first_log:
                AudioSignal(audio, sample_rate=self.trainer.datamodule.sample_rate).write_audio_to_tb(
                    f"gt/{i}.wav", writer, self.global_step
                )
            audio = audio.unsqueeze(0).to(self.device)
            pred = self.generator(audio).float()
            signal = AudioSignal(pred, sample_rate=self.trainer.datamodule.sample_rate)
            signal = signal.ensure_max_of_audio().cpu()
            signal.write_audio_to_tb(f"gen/{i}.wav", writer, self.global_step)
        self.first_log = False
        self.generator.train()

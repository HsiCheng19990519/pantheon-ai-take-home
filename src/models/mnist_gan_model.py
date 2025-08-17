from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()

        # Keep LSGAN (MSE) as default for backward compatibility.
        self.loss_type = getattr(self.hparams, "loss_type", "lsgan")

        # EMA settings (for better sampling stability). Disabled by default.
        self.ema_enabled = bool(getattr(self.hparams, "ema_enabled", False))
        self.ema_decay = float(getattr(self.hparams, "ema_decay", 0.999))

        if self.ema_enabled:
            import copy
            # Create a non-trainable shadow copy of the generator
            self.generator_ema = copy.deepcopy(self.generator)
            for p in self.generator_ema.parameters():
                p.requires_grad = False

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        # TTUR: allow different lr/betas for G and D; fall back to single lr/betas if not provided.
        lr_g = float(getattr(self.hparams, "lr_g", 0.0)) or float(self.hparams.lr)
        lr_d = float(getattr(self.hparams, "lr_d", 0.0)) or float(self.hparams.lr)

        b1_g = float(getattr(self.hparams, "b1_g", 0.0)) or float(self.hparams.b1)
        b2_g = float(getattr(self.hparams, "b2_g", 0.0)) or float(self.hparams.b2)
        b1_d = float(getattr(self.hparams, "b1_d", 0.0)) or float(self.hparams.b1)
        b2_d = float(getattr(self.hparams, "b2_d", 0.0)) or float(self.hparams.b2)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1_g, b2_g))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(b1_d, b2_d))
        return [opt_g, opt_d], []


    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        if self.ema_enabled and optimizer_idx == 0:
            # flag so that we update EMA *after* the optimizer step
            self._did_g_step = True
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        # Test validation: run a forward-like step and log metrics.
        log_dict, _ = self.step(batch, batch_idx)
        self.log_dict({"/".join(("test", k)): v for k, v in log_dict.items()})
        return None

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Update EMA right after optimizer steps of G.
        if getattr(self, "_did_g_step", False) and self.ema_enabled:
            with torch.no_grad():
                d = float(self.ema_decay)
                for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
                    p_ema.copy_(p_ema * d + p.detach() * (1.0 - d))
                # synchronize buffers (e.g., BatchNorm running stats) to keep sampling stable
                for b_ema, b in zip(self.generator_ema.buffers(), self.generator.buffers()):
                    b_ema.copy_(b)
        self._did_g_step = False


    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        # TODO: implement the step method of the GAN model.
        #     : This function should return both a dictionary of losses
        #     : and current loss of the network being optimised.
        #     :
        #     : When training with pytorch lightning, because we defined 2 optimizers in
        #     : the `configure_optimizers` function above, we use the `optimizer_idx` parameter
        #     : to keep a track of which network is being optimised.

        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # TODO: Create adversarial ground truths
        # For LSGAN (MSELoss): real targets are 1, fake targets are 0.
        # Shape must match the discriminator's output: [B, 1].
        valid = torch.ones((batch_size, 1), device=imgs.device, dtype=imgs.dtype)
        fake = torch.zeros((batch_size, 1), device=imgs.device, dtype=imgs.dtype)

        # TODO: Create noise and labels for generator input
        # Sample Gaussian noise z ~ N(0, I) and random class labels for conditional generation.
        z = torch.randn(batch_size, self.hparams.latent_dim, device=imgs.device)
        gen_labels = torch.randint(low=0, high=self.hparams.n_classes, size=(batch_size,), device=imgs.device)

        if optimizer_idx == 0 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the generator
            # HINT: when optimizer_idx == 0 the model is optimizing the generator

            # TODO: Generate a batch of images
            # G wants D(G(z)) -> 1 (valid), so minimize MSE(D(G(z)), 1)
            gen_imgs = self.generator(z, gen_labels)
            pred_fake = self.discriminator(gen_imgs, gen_labels)
            # TODO: Calculate loss to measure generator's ability to fool the discriminator
            if self.loss_type == "hinge":
                # Generator wants D(fake) to be large; hinge-G loss = -E[D(fake)]
                g_loss = -pred_fake.mean()
            else:
                # LSGAN (MSE) as before
                g_loss = self.adversarial_loss(pred_fake, valid)

            log_dict.update({
                "g_loss": g_loss.detach(),
                "d_pred_fake_mean": pred_fake.detach().mean(),
            })
            # Only set `loss` when actually optimizing G (i.e., in training with optimizer_idx==0)
            if optimizer_idx == 0:
                loss = g_loss

        if optimizer_idx == 1 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the discriminator
            # HINT: when optimizer_idx == 1 the model is optimizing the discriminator

            # D wants D(real) -> 1 and D(fake) -> 0.
            # TODO: Generate a batch of images
            # Use detached fake samples to avoid backprop through G when optimizing D.
            with torch.no_grad():
                gen_imgs_for_d = self.generator(z, gen_labels)
            pred_real = self.discriminator(imgs, labels)
            pred_fake_detached = self.discriminator(gen_imgs_for_d, gen_labels)

            # TODO: Calculate loss for real images
            if self.loss_type == "hinge":
                loss_real = torch.relu(1.0 - pred_real).mean()
            else:
                loss_real = self.adversarial_loss(pred_real, valid)

            # TODO: Calculate loss for fake images
            if self.loss_type == "hinge":
                loss_fake = torch.relu(1.0 + pred_fake_detached).mean()
            else:
                loss_fake = self.adversarial_loss(pred_fake_detached, fake)

            # TODO: Calculate total discriminator loss
            d_loss = 0.5 * (loss_real + loss_fake)

            log_dict.update({
                "d_loss": d_loss.detach(),
                "d_loss_real": loss_real.detach(),
                "d_loss_fake": loss_fake.detach(),
                "d_pred_real_mean": pred_real.detach().mean(),
                "d_pred_fake_detached_mean": pred_fake_detached.detach().mean(),
            })
            if optimizer_idx == 1:
                loss = d_loss

        return log_dict, loss

    def on_epoch_end(self):
        # TODO: implement functionality to log predicted images to wandb
        #     : at the end of each epoch

        # TODO: Create fake images
        # Sample a small, fixed-size batch for qualitative monitoring.
        import torchvision.utils as vutils  # local import to keep file-level imports unchanged

        self.generator.eval()
        with torch.no_grad():
            num_samples = 25  # 5x5 grid
            z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
            sample_labels = torch.randint(
                low=0, high=self.hparams.n_classes, size=(num_samples,), device=self.device
            )
            gen_imgs = self.generator(z, sample_labels)  # in [-1, 1] due to Tanh

            # Make a nice grid and normalize to [0,1] for visualization
            grid = vutils.make_grid(gen_imgs, nrow=5, normalize=True, value_range=(-1, 1))

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # TODO: log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)
                #     : replace `None` with your wandb Image object
                logger.experiment.log({"gen_imgs": wandb.Image(grid, caption=f"epoch={self.current_epoch}")})
                
        # Optionally log EMA samples for a side-by-side comparison (more stable visuals)
        if getattr(self, "ema_enabled", False):
            self.generator_ema.eval()
            with torch.no_grad():
                gen_imgs_ema = self.generator_ema(z, sample_labels)
                grid_ema = vutils.make_grid(gen_imgs_ema, nrow=5, normalize=True, value_range=(-1, 1))
            for logger in self.trainer.logger:
                if type(logger).__name__ == "WandbLogger":
                    logger.experiment.log({
                        "gen_imgs_ema": wandb.Image(grid_ema, caption=f"epoch={self.current_epoch} (EMA)")
                    })


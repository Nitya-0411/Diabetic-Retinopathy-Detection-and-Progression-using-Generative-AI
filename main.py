import os
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Custom imports for CycleGAN architecture
from models.generator import Generator
from models.discriminator import Discriminator
from utils.buffer import ReplayBuffer
from utils.lambda_lr import LambdaLR
from utils.weight_init import weights_init_normal
from dataset.image_dataset import ImageDataset

# Configuration
class Config:
    def __init__(self, source_stage, target_stage):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.lr = 0.0002
        self.n_epochs = 200
        self.decay_epoch = 100
        self.size = 256
        self.input_nc = 3
        self.output_nc = 3
        self.source_stage = source_stage
        self.target_stage = target_stage
        self.dataroot = f"/content/drive/MyDrive/Project/GenAI/{source_stage}_{target_stage}"
        self.save_dir = f"models/{source_stage}_to_{target_stage}"
        self.lambda_identity = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0

        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(f"runs/{source_stage}_to_{target_stage}")

def train_cyclegan(source_stage, target_stage):
    opt = Config(source_stage, target_stage)
    print(f"🔄 Training CycleGAN for {source_stage} → {target_stage} on {opt.device}")

    # Initialize models
    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(opt.device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(opt.device)
    netD_A = Discriminator(opt.input_nc).to(opt.device)
    netD_B = Discriminator(opt.output_nc).to(opt.device)

    # Initialize weights
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=opt.lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # LR schedulers
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step
    )

    # Buffers
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Transform: minimal preprocessing
    transform = transforms.Compose([
        transforms.Resize((opt.size, opt.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset and DataLoader
    dataset = ImageDataset(opt.dataroot, transform=transform, unaligned=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(opt.device)
            real_B = batch['B'].to(opt.device)

            # -------------------
            #  Train Generators
            # -------------------
            optimizer_G.zero_grad()

            # Identity loss
            loss_identity_B = criterion_identity(netG_A2B(real_B), real_B) * opt.lambda_B * opt.lambda_identity
            loss_identity_A = criterion_identity(netG_B2A(real_A), real_A) * opt.lambda_A * opt.lambda_identity

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            valid_B = torch.ones_like(pred_fake_B, device=opt.device)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, valid_B)

            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            valid_A = torch.ones_like(pred_fake_A, device=opt.device)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, valid_A)

            # Cycle loss
            loss_cycle_ABA = criterion_cycle(netG_B2A(fake_B), real_A) * opt.lambda_A
            loss_cycle_BAB = criterion_cycle(netG_A2B(fake_A), real_B) * opt.lambda_B

            # Total generator loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            optimizer_D_A.zero_grad()
            pred_real_A = netD_A(real_A)
            valid_A = torch.ones_like(pred_real_A, device=opt.device)
            loss_D_real = criterion_GAN(pred_real_A, valid_A)

            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = netD_A(fake_A_.detach())
            fake_label_A = torch.zeros_like(pred_fake_A, device=opt.device)
            loss_D_fake = criterion_GAN(pred_fake_A, fake_label_A)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            optimizer_D_B.zero_grad()
            pred_real_B = netD_B(real_B)
            valid_B = torch.ones_like(pred_real_B, device=opt.device)
            loss_D_real = criterion_GAN(pred_real_B, valid_B)

            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = netD_B(fake_B_.detach())
            fake_label_B = torch.zeros_like(pred_fake_B, device=opt.device)
            loss_D_fake = criterion_GAN(pred_fake_B, fake_label_B)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # Logging
            if i % 25 == 0:
                step = epoch * len(dataloader) + i
                print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D_A: {loss_D_A.item():.4f}] [D_B: {loss_D_B.item():.4f}] [G: {loss_G.item():.4f}]")

                opt.writer.add_scalar('loss_G', loss_G.item(), step)
                opt.writer.add_scalar('loss_D_A', loss_D_A.item(), step)
                opt.writer.add_scalar('loss_D_B', loss_D_B.item(), step)

                if i % 100 == 0:
                    images = torch.cat([
                        real_A.data, fake_B.data, netG_B2A(fake_B).data,
                        real_B.data, fake_A.data, netG_A2B(fake_A).data
                    ], 0)
                    grid = vutils.make_grid(images, normalize=True, nrow=3)
                    opt.writer.add_image('CycleGAN_samples', grid, step)

        # Update LR
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == opt.n_epochs - 1:
            torch.save(netG_A2B.state_dict(), f"{opt.save_dir}/netG_A2B_epoch_{epoch}.pth")
            torch.save(netG_B2A.state_dict(), f"{opt.save_dir}/netG_B2A_epoch_{epoch}.pth")
            torch.save(netD_A.state_dict(), f"{opt.save_dir}/netD_A_epoch_{epoch}.pth")
            torch.save(netD_B.state_dict(), f"{opt.save_dir}/netD_B_epoch_{epoch}.pth")

    # Final save
    torch.save(netG_A2B.state_dict(), f"{opt.save_dir}/netG_A2B_final.pth")
    torch.save(netG_B2A.state_dict(), f"{opt.save_dir}/netG_B2A_final.pth")
    print(f"✅ Training complete for {source_stage} → {target_stage}")

# Entry point
if __name__ == "__main__":
    train_cyclegan("Mild", "Moderate")
    train_cyclegan("Moderate", "Severe")

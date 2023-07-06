import torch
import pickle
import numpy as np
from tqdm import tqdm
from ganModel import Generator, Discriminator
from config import Config
import torch.autograd as autograd

trainData = pickle.load(open('/data/theodoroubp/imageGen/trainData.pkl', 'rb')) + pickle.load(open('/data/theodoroubp/imageGen/valData.pkl', 'rb'))
pretrainData = [d for d in trainData]
trainData = [d for d in trainData if d[2] is not None]

config = Config()
EPOCHS = config.gan_epochs
PATIENCE = config.patience
LR = config.lr
BATCH_SIZE = config.batch_size
NOISE_STEPS = config.noise_steps
EMBED_DIM = config.embed_dim
LATENT_DIM = config.latent_dim
LAMBDA_GP = config.lambda_gp
GENERATOR_INTERVAL = config.generator_interval * BATCH_SIZE
SAVE_INTERVAL = config.save_interval * BATCH_SIZE
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIM = len(trainData[0][1])
COND_DIM = len(trainData[0][0]) + IMAGE_DIM

def get_prebatch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    condData = torch.zeros(bs, COND_DIM)
    imageData = torch.zeros(bs, IMAGE_DIM)
    for i, d in enumerate(data):
        imageData[i] = d[1]

    return imageData.to(DEVICE), condData.to(DEVICE)

def get_batch(dataset, loc, batch_size):
    data = dataset[loc:loc+batch_size]
    bs = len(data)
    condData = torch.zeros(bs, COND_DIM)
    imageData = torch.zeros(bs, IMAGE_DIM)
    for i, d in enumerate(data):
        imageData[i] = d[2]
        condData[i,:-IMAGE_DIM] = d[0]
        condData[i,-IMAGE_DIM:] = d[1]

    return imageData.to(DEVICE), condData.to(DEVICE)

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

generator = Generator(IMAGE_DIM, LATENT_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
discriminator = Discriminator(IMAGE_DIM, embed_dim=EMBED_DIM, condition=True, cond_dim=COND_DIM).to(DEVICE)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR)

for e in range(EPOCHS):
    np.random.shuffle(pretrainData)
    generator.train()
    discriminator.train()
    for i in tqdm(range(0, len(pretrainData), BATCH_SIZE), leave=False):
        real_imgs, condData = get_batch(pretrainData, i, BATCH_SIZE)

        ###########################
        ### Train Discriminator ###
        ###########################

        # Generate a batch of images
        z = torch.randn(real_imgs.size(0), LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z, condData)

        # Real images
        real_validity = discriminator(real_imgs, condData)
        # Fake images
        fake_validity = discriminator(fake_imgs, condData)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, condData.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        if i % GENERATOR_INTERVAL == 0:
            #######################
            ### Train Generator ###
            #######################

            # Generate a batch of images
            fake_imgs = generator(z, condData)

            # Loss measures generator's ability to fool the discriminator
            fake_validity = discriminator(fake_imgs, condData)
            g_loss = -torch.mean(fake_validity)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (e, EPOCHS, i, len(pretrainData), d_loss.detach().cpu().item(), g_loss.detach().cpu().item())
            )

            if i % SAVE_INTERVAL == 0:
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optimizer_G': optimizer_G,
                    'optimizer_D': optimizer_D,
                    'epoch': e,
                    'mode': 'pretrain'
                }, '/data/theodoroubp/imageGen/save/pretrained_gan_model')

for e in range(EPOCHS):
    np.random.shuffle(trainData)
    generator.train()
    discriminator.train()
    for i in tqdm(range(0, len(trainData), BATCH_SIZE), leave=False):
        real_imgs, condData = get_batch(trainData, i, BATCH_SIZE)

        ###########################
        ### Train Discriminator ###
        ###########################

        # Generate a batch of images
        z = torch.randn(real_imgs.size(0), LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z, condData)

        # Real images
        real_validity = discriminator(real_imgs, condData)
        # Fake images
        fake_validity = discriminator(fake_imgs, condData)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, condData.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        if i % GENERATOR_INTERVAL == 0:
            #######################
            ### Train Generator ###
            #######################

            # Generate a batch of images
            fake_imgs = generator(z, condData)

            # Loss measures generator's ability to fool the discriminator
            fake_validity = discriminator(fake_imgs, condData)
            g_loss = -torch.mean(fake_validity)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (e, EPOCHS, i, len(trainData), d_loss.detach().cpu().item(), g_loss.detach().cpu().item())
            )

            if i % SAVE_INTERVAL == 0:
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'optimizer_G': optimizer_G,
                    'optimizer_D': optimizer_D,
                    'epoch': e,
                    'mode': 'train'
                }, '/data/theodoroubp/imageGen/save/pretrained_gan_model')
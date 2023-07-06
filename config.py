class Config:
    def __init__(self,
        epochs = 1000,
        patience = 100,
        lr = 1e-3,
        batch_size = 256,
        noise_steps = 100,
        embed_dim = 256,
        beta_start = 1e-4,
        beta_end = 0.02,
        gan_epochs = 200,
        latent_dim = 100,
        lambda_gp = 10,
        generator_interval = 5,
        save_interval = 10,
    ):
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.noise_steps = noise_steps
        self.embed_dim = embed_dim
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gan_epochs = gan_epochs
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.generator_interval = generator_interval
        self.save_interval = save_interval
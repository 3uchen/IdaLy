import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import scipy


class Gan_generator(nn.Module):
    def __init__(self, latent_dim=100, data_dim=52):
        super(Gan_generator, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 64, normalize=False),
            *block(64, 32),
            nn.Linear(32, self.data_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Gan_discriminator(nn.Module):
    def __init__(self, data_dim=100):
        super(Gan_discriminator, self).__init__()
        self.data_dim = data_dim

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class Vae(nn.Module):
    def __init__(self, data_dim, latent_dim) -> None:
        super(Vae, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
    
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.encoder = nn.Sequential(
            # *block(self.data_dim, 256),
            *block(self.data_dim, 64),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        self.mean_linear = nn.Linear(32, self.latent_dim)
        self.var_linear = nn.Linear(32, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            # nn.Linear(128, 64),
            # nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, self.data_dim),
            nn.Sigmoid(),
        )
  
    def forward(self, x):
        encode = self.encoder(x)
        mean = self.mean_linear(encode)
        var = self.var_linear(encode)
        eps = torch.randn_like(var)
        std = torch.exp(var/2)
        z = eps*std + mean
        reconstrustion = self.decoder(z)
        return reconstrustion, mean, var


class WGan_generator(nn.Module):
    def __init__(self, latent_dim=100, data_dim=52):
        super(WGan_generator, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 64, normalize=False),
            *block(64, 32),
            nn.Linear(32, self.data_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class WGan_discriminator(nn.Module):
    def __init__(self, data_dim=100):
        super(WGan_discriminator, self).__init__()
        self.data_dim = data_dim

        self.model = nn.Sequential(
            nn.Linear(self.data_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class BasicUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.ConvTranspose1d(64, 64, kernel_size=5, padding=2),
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2),
            nn.ConvTranspose1d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool1d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2: 
              h.append(x) 
              x = self.downscale(x) 
              
        for i, l in enumerate(self.up_layers):
            if i > 0: 
              x = self.upscale(x)
              x += h.pop() 
            x = self.act(l(x))          
        return x


class Vaegan_encoderblock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Vaegan_encoderblock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv1d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(num_features=channel_out, momentum=0.9)
    
    def forward(self, ten):
        ten_1 = self.conv(ten.clone())
        ten_2 = self.bn(ten_1.clone())
        ten_3 = F.relu(ten_2.clone(), True)
        return ten_3
    

class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose1d(channel_in, channel_out, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten_1 = self.conv(ten.clone())
        ten_2 = self.bn(ten_1.clone())
        ten_3 = F.relu(ten_2.clone(), True)
        return ten_3
    

class Vaegan_Encoder(nn.Module):
    def __init__(self, data_dim, channel_in=1, z_size=128):
        super(Vaegan_Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        for i in range(2):
            if i == 0:
                layers_list.append(Vaegan_encoderblock(channel_in=self.size, channel_out=16))
                self.size = 16
            else:
                layers_list.append(Vaegan_encoderblock(channel_in=self.size, channel_out=self.size * 2))
                self.size = self.size*2

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=data_dim*self.size, out_features=64, bias=False),
                                nn.BatchNorm1d(num_features=64, momentum=0.9),
                                nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=64, out_features=z_size)
        self.l_var = nn.Linear(in_features=64, out_features=z_size)

    def forward(self, ten):
        ten_1 = self.conv(ten.clone())
        ten_2 = ten_1.view(len(ten_1), -1)
        ten_3 = self.fc(ten_2.clone())
        mu = self.l_mu(ten_3)
        logvar = self.l_var(ten_3)
        kl = -0.5 * torch.sum(-logvar.exp() - torch.pow(mu, 2) + logvar + 1, 1)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar/2)
        z = eps*std + mu
        return z, kl


class Vaegan_Decoder(nn.Module):
    def __init__(self, data_dim, z_size, size=32):
        super(Vaegan_Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=data_dim*size, bias=False),
                                nn.BatchNorm1d(num_features=data_dim*size, momentum=0.9),
                                nn.ReLU(True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//2))
        self.size = self.size//2
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.size, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))
        self.data_dim = data_dim
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten_1 = self.fc(ten.clone())
        ten_2 = ten_1.clone().view(len(ten_1), -1, self.data_dim)
        ten_3 = self.conv(ten_2.clone())
        return ten_3


class Vaegan_Discriminator(nn.Module):
    def __init__(self, data_dim, channel_in=1, recon_level=3):
        super(Vaegan_Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.conv.append(Vaegan_encoderblock(channel_in=16, channel_out=32))
        self.conv.append(Vaegan_encoderblock(channel_in=32, channel_out=64))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=data_dim*64, out_features=64, bias=False),
            nn.BatchNorm1d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, ten, mode='REC'):
        if mode == "REC":
            for i, lay in enumerate(self.conv):
                ten = lay(ten.clone())
            return ten.clone().view(len(ten), -1)
        else:
            for i, lay in enumerate(self.conv):
                ten = lay(ten.clone())
            ten = ten.clone().view(len(ten), -1)
            ten = self.fc(ten.clone())
            return F.sigmoid(ten)
        

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
    

class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = scipy.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))
        self.I_ = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I_ = self.I_.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I_
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)

class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs
            
        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask    
        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

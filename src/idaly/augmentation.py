#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：3uchen time:2022/12/5
import random
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
from scipy.linalg import solve
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
import model
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from torch import manual_seed, cuda
import torch.nn as nn
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis


def pca(ori_data, gen_data):
    all_data = np.concatenate((ori_data, gen_data), axis=0)
    p = PCA(n_components=2)
    pca_2d = p.fit_transform(all_data)
    p_ori = pca_2d[:len(ori_data)]
    p_gen = pca_2d[len(ori_data):]
    np.random.shuffle(p_gen)
    np.random.shuffle(p_ori)
    pca_data = [p_ori[:50], p_gen[:50]]
    return pca_data


def setup_seed(seed):
    manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class GNI:

    def __init__(self, num_gen, mean, variance):
        self.mean = mean
        self.variance = variance
        self.num_gen = num_gen

    def fit(self, original_data):
        if self.num_gen <= original_data.shape[0]:
            X_f = original_data[:self.num_gen, :]
        else:
            repeats = int(self.num_gen / original_data.shape[0])
            yu = self.num_gen % original_data.shape[0]
            X_f = np.repeat(original_data, repeats, axis=0)
            X_f = np.concatenate((X_f, original_data[:yu, :]), axis=0)
        noise = np.random.normal(loc=self.mean,
                                 scale=self.variance,
                                 size=(X_f.shape[0], X_f.shape[1]))
        GNI_gen = X_f + noise 
        return GNI_gen
    

class SNI:

    def __init__(self, num_gen, mean, variance):
        self.mean = mean
        self.variance = variance
        self.num_gen = num_gen

    def fit(self, original_data):
        if self.num_gen <= original_data.shape[0]:
            X_f = original_data[:self.num_gen, :]
        else:
            repeats = int(self.num_gen / original_data.shape[0])
            yu = self.num_gen % original_data.shape[0]
            X_f = np.repeat(original_data, repeats, axis=0)
            X_f = np.concatenate((X_f, original_data[:yu, :]), axis=0)
        noise = np.random.normal(loc=self.mean,
                                 scale=self.variance,
                                 size=(X_f.shape[0], X_f.shape[1]))
        SNI_gen = X_f + noise * X_f
        return SNI_gen
    

class PNI:

    def __init__(self, num_gen):
        self.num_gen = num_gen

    def fit(self, original_data):
        if self.num_gen <= original_data.shape[0]:
            X_f = original_data[:self.num_gen, :]
        else:
            repeats = int(self.num_gen / original_data.shape[0])
            yu = self.num_gen % original_data.shape[0]
            X_f = np.repeat(original_data, repeats, axis=0)
            X_f = np.concatenate((X_f, original_data[:yu, :]), axis=0)
        vals = len(np.unique(X_f))
        vals = 2 ** np.ceil(np.log2(vals))
        PNI_gen = np.random.poisson(X_f * vals) / float(vals)
        return PNI_gen


class MSI:

    def __init__(self, num_gen, mask_prob):
        self.mask_prob = mask_prob
        self.num_gen = num_gen

    def fit(self, original_data):
        if self.num_gen <= original_data.shape[0]:
            MSI_gen = original_data[:self.num_gen, :]
        else:
            repeats = int(self.num_gen / original_data.shape[0])
            yu = self.num_gen % original_data.shape[0]
            X_f = np.repeat(original_data, repeats, axis=0)
            MSI_gen = np.concatenate((X_f, original_data[:yu, :]), axis=0)
        mask = np.random.random(MSI_gen.shape) < self.mask_prob  
        MSI_gen[mask] = 0
        return MSI_gen

class Cutout:
    def __init__(self, num_gen, mask_prob):
        self.mask_prob = mask_prob
        self.num_gen = num_gen
    
    def generate_mask(self, shape):
        mask = np.ones(shape, dtype=int)
        for i in range(shape[0]):
            L = int(self.mask_prob * shape[1])
            max_start = shape[1] - L
            start = random.randint(0, max_start)
            mask[i, start : start + L] = 0
        return mask


    def fit(self, original_data):
        if self.num_gen <= original_data.shape[0]:
            MSI_gen = original_data[:self.num_gen, :]
        else:
            repeats = int(self.num_gen / original_data.shape[0])
            yu = self.num_gen % original_data.shape[0]
            X_f = np.repeat(original_data, repeats, axis=0)
            MSI_gen = np.concatenate((X_f, original_data[:yu, :]), axis=0)
        mask = self.generate_mask(MSI_gen.shape) 
        MSI_gen[mask] = 0
        return MSI_gen

class Smote(object):

    def __init__(self, N=100, k=10, r=2):
        self.N = N
        self.k = k
        self.r = r
        self.newindex = 0

    def fit(self, samples):
        T = samples.shape[0]
        numattrs = samples.shape[1]
        self.synthetic = np.zeros((self.N, numattrs))
        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(samples)
        repeats = int(self.N / T)
        yu = self.N % T
        if repeats > 0:
            for i in range(T):
                nnarray = neighbors.kneighbors(samples[i].reshape((1, -1)),
                                               return_distance=False)[0][1:]
                self._populate(repeats, i, nnarray, samples)

        samples_shuffle = samples
        np.random.shuffle(samples_shuffle)
        neighbors_1 = NearestNeighbors(n_neighbors=self.k + 1,
                                       algorithm='ball_tree',
                                       p=self.r).fit(samples_shuffle)
        for j in range(yu):
            nnarray_1 = neighbors_1.kneighbors(samples_shuffle[j].reshape(
                (1, -1)),
                                               return_distance=False)[0][1:]
            self._populate(1, j, nnarray_1, samples_shuffle)
        return self.synthetic

    def _populate(self, N, i, nnarray, samples):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            diff = samples[nnarray[nn]] - samples[i]
            gap = random.uniform(0, 1)
            self.synthetic[self.newindex] = samples[i] + gap * diff
            self.newindex += 1


class Kmeans_Smote:
    def __init__(self, num_gen, n_clusters):
        self.n_clusters = n_clusters
        self.num_gen = num_gen

    def fit(self, samples):
        cluster = KMeans(n_clusters=self.n_clusters).fit(samples)
        labels = np.array(cluster.labels_)
        cluster_index = []
        for i in range(self.n_clusters):
            cluster_i = np.where(labels == i)
            cluster_i = np.array(cluster_i)
            cluster_index.append(cluster_i[0])
        centers = cluster.cluster_centers_
        gen_data = np.zeros((self.num_gen, samples.shape[1]))
        num = 0
        while num < self.num_gen:
            for i in range(self.n_clusters):
                idex = cluster_index[i][np.random.randint(0, len(cluster_index[i]))]
                sample = samples[idex]
                synthetic_sample = centers[i] + np.random.rand() * (sample - centers[i])
                gen_data[num] = synthetic_sample
                num = num + 1
                if num == self.num_gen:
                    break    
        return gen_data


class Mixup:
    def __init__(self, num_gen, alpha):
        self.num_gen = num_gen
        self.alpha = alpha

    def fit(self, samples):
        np.random.shuffle(samples)
        gen_samples = np.zeros((self.num_gen, samples.shape[1]))
        for i in range(self.num_gen):
            m = random.randint(0, samples.shape[0] - 1)
            while True:
                n = random.randint(0, samples.shape[0] - 1)
                if m != n:
                    break
            lam = np.random.beta(self.alpha, self.alpha)
            mix = lam * samples[m] + (1 - lam) * samples[n]
            gen_samples[i] = mix
        return gen_samples


class Lle:
    def __init__(self, num_gen, n_neighbor, reg, n_component):
        self.num_gen = num_gen
        self.k = n_neighbor
        self.reg = reg
        self.n_components = n_component

    def barycenter_weights(self, x, y, indices):
        x = check_array(x, dtype=FLOAT_DTYPES)
        y = check_array(y, dtype=FLOAT_DTYPES)
        indices = check_array(indices, dtype=int)

        n_samples, n_neighbors = indices.shape
        assert x.shape[0] == n_samples

        B = np.empty((n_samples, n_neighbors), dtype=x.dtype)
        v = np.ones(n_neighbors, dtype=x.dtype)

        for i, ind in enumerate(indices):
            A = y[ind]
            C = A - x[i]  # broadcasting
            G = np.dot(C, C.T)
            trace = np.trace(G)
            if trace > 0:
                R = self.reg * trace
            else:
                R = self.reg
            G.flat[::n_neighbors + 1] += R
            w = solve(G, v, sym_pos=True)
            B[i, :] = w / np.sum(w)
        return B

    def reconstruct(self, x_vir_low, x_low, x_train):
        x = np.vstack((x_vir_low, x_low))
        knn = NearestNeighbors(n_neighbors=self.k + 1).fit(x)
        X = knn._fit_X
        ind = knn.kneighbors(X, return_distance=False)[:, 1:]
        w = self.barycenter_weights(X, X, ind)
        x_vir = np.dot(w[0], x_train[ind - 1])
        return x_vir[0]

    def random_sample(self, x_low, nums, n_components):
        x_min = np.min(x_low, 0)
        x_max = np.max(x_low, 0)
        z = np.random.rand(nums, n_components)
        x_vir_lows = z * (x_max - x_min) + x_min
        return x_vir_lows

    def fit(self, samples):
        res = np.zeros((self.num_gen, samples.shape[1]))
        x_low = manifold.LocallyLinearEmbedding(
            n_neighbors=self.k,
            n_components=self.n_components,
            method='standard').fit_transform(samples)
        x_vir = self.random_sample(x_low, self.num_gen, self.n_components)
        for i in range(self.num_gen):
            a = np.array([x_vir[i]])
            res[i] = self.reconstruct(a, x_low, samples)
        return res


class MTD:

    def __init__(self, n_obs=100, random_state=8):
        self.n_obs = n_obs
        self._gen_obs = n_obs * 20
        self.synthetic = None
        np.random.RandomState(random_state)

    def diffusion(self, sample):
        new_sample = []
        min_val = np.min(sample)
        max_val = np.max(sample)
        u_set = (min_val + max_val) / 2
        if u_set == min_val or u_set == max_val:
            Nl = len([i for i in sample if i <= u_set])
            Nu = len([i for i in sample if i >= u_set])
        else:
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
        skew_l = Nl / (Nl + Nu)
        skew_u = Nu / (Nl + Nu)
        var = np.var(sample, ddof=1)
        if var == 0:
            a = min_val / 5
            b = max_val * 5
            new_sample = np.random.uniform(a, b, size=self._gen_obs)
        else:
            a = u_set - (skew_l * np.sqrt(-2 * (var / Nl) * np.log(10**(-20))))
            b = u_set + (skew_u * np.sqrt(-2 * (var / Nu) * np.log(10**(-20))))
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            while len(new_sample) < self._gen_obs:
                x = np.random.uniform(L, U)
                if x <= u_set:
                    MF = (x - L) / (u_set - L)
                elif x > u_set:
                    MF = (U - x) / (U - u_set)
                elif x < L or x > U:
                    MF = 0
                rs = np.random.uniform(0, 1)
                # new_sample.append(x)
                if MF > rs:
                    new_sample.append(x)
                else:
                    continue
        return np.array(new_sample)

    def fit(self, original_data):
        samples = original_data
        numattrs = samples.shape[1]
        temp = np.zeros((self._gen_obs, numattrs))
        for col in range(numattrs):
            y = samples[:, col]
            diff_out = self.diffusion(y)
            temp[:, col] = diff_out
        np.random.shuffle(temp)
        self.synthetic = temp[:self.n_obs]
        return self.synthetic


class kNNMTD(MTD):
    """
       Reference:
                @article{sivakumar2021synthetic,
                         title={Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors},
                         author={Sivakumar, Jayanth and Ramamurthy, Karthik and Radhakrishnan, Menaka and Won, Daehan},
                         journal={Knowledge-Based Systems},
                         pages={107687},
                         year={2021},
                         publisher={Elsevier}
                        }
    """
    def __init__(self, n_obs=100, k=3, random_state=42):
        self.n_obs = n_obs
        self._gen_obs = k * 10
        self.k = k
        self.synthetic = None
        np.random.RandomState(random_state)

    def getNeighbors(self, val, x):
        dis_set = []
        for m in x:
            dis = np.abs(m - val)
            dis_set.append(dis)
        dist_array = np.array(dis_set).reshape(1, -1)[0]
        indx = np.argsort(dist_array)
        k_nei = x[indx][:self.k]
        return k_nei

    def fit(self, original_data):
        samples = original_data
        numattrs = samples.shape[1]
        M = 0
        while M < self.n_obs:
            T = samples.shape[0]
            temp = np.zeros((self.k * T, numattrs))
            for row in range(T):
                val = samples[row]
                for col in range(numattrs):
                    y = samples[:, col].reshape(-1, 1)
                    neighbor_df = self.getNeighbors(val[col], y)
                    diff_out = self.diffusion(neighbor_df)
                    k_col = self.getNeighbors(val[col], diff_out)
                    temp[row * self.k:(row + 1) * self.k, col] = k_col
            samples = np.concatenate([samples, temp], axis=0)
            np.random.shuffle(samples)
            M = temp.shape[0]
        np.random.shuffle(temp)
        self.synthetic = temp[:self.n_obs]
        return self.synthetic
    

class MD_MTD:
    def __init__(self, n_obs=100, random_state=8):
        self.n_obs = n_obs
        self._gen_obs = n_obs * 20
        self.synthetic = None
        np.random.RandomState(random_state)

    def diffusion(self, sample):
        new_sample = []
        mean = np.mean(sample)
        sigma = np.std(sample)
        sampled = len([i for i in sample if np.abs(i - mean) < 3 * sigma])
        min_val = np.min(sample)
        max_val = np.max(sample)
        # sorted_sample = sorted(sample)
        # if len(sample) % 2 == 0:
        #     u_set = (sorted_sample[int(len(sample)/2)] + sorted_sample[int(len(sample)/2 + 1)]) / 2
        # else:
        #     u_set = sorted_sample[int(len(sample)/2 + 1/2)]
        u_set = (min_val + max_val) / 2
        if u_set == min_val or u_set == max_val:
            Nl = len([i for i in sample if i <= u_set])
            Nu = len([i for i in sample if i >= u_set])
        else:
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
        skew_l = Nl / (Nl + Nu)
        skew_u = Nu / (Nl + Nu)
        var = np.var(sample, ddof=1)
        if var == 0:
            a = min_val / 5
            b = max_val * 5
            new_sample = np.random.uniform(a, b, size=self._gen_obs)
        else:
            a = u_set - (skew_l * np.sqrt(-2 * (var / Nl) * np.log(10**(-20))))
            b = u_set + (skew_u * np.sqrt(-2 * (var / Nu) * np.log(10**(-20))))
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            print(min_val, a, max_val, b)
            while len(new_sample) < self._gen_obs:
                x = np.random.uniform(L, U)
                if x <= u_set:
                    MF = (x - L) / (u_set - L)
                elif x > u_set:
                    MF = (U - x) / (U - u_set)
                elif x < L or x > U:
                    MF = 0
                rs = np.random.uniform(0, 1)
                # new_sample.append(x)
                if MF > rs:
                    new_sample.append(x)
                else:
                    continue
        return np.array(new_sample)

    def fit(self, original_data):
        samples = original_data
        numattrs = samples.shape[1]
        temp = np.zeros((self._gen_obs, numattrs))
        for col in range(numattrs):
            y = samples[:, col]
            diff_out = self.diffusion(y)
            temp[:, col] = diff_out
        np.random.shuffle(temp)
        self.synthetic = temp[:self.n_obs]
        return self.synthetic



class Gmm(object):

    def __init__(self, N, n_components):
        self.N = N
        self.n_components = n_components

    def fit(self, samples):
        gmm = GMM(n_components=self.n_components)
        gmm.fit(samples)
        print("hhhh")
        self.synthetic = gmm.sample(self.N)[0]
        return self.synthetic


class GAN:

    def __init__(self, num_gen, num_epoch, lr, batch_size, latent_dim):
        self.num_gen = num_gen
        self.num_epoch = num_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.cuda = True if cuda.is_available() else False
        self.setup_seed(42)

    def setup_seed(self, seed):
        manual_seed(seed)
        cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def train(self, train_data):
        if self.cuda:
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        data_dim = train_data.shape[1]
        if self.cuda:
            netG = model.Gan_generator(latent_dim=self.latent_dim,
                                       data_dim=data_dim).cuda()
            netD = model.Gan_discriminator(data_dim=data_dim).cuda()
        else:
            netG = model.Gan_generator(latent_dim=self.latent_dim,
                                       data_dim=data_dim)
            netD = model.Gan_discriminator(data_dim=data_dim)
        dataset = torch.tensor(train_data)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        optimizer_G = torch.optim.Adam(netG.parameters(),
                                       lr=self.lr,
                                       betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(netD.parameters(),
                                       lr=self.lr,
                                       betas=(0.5, 0.999))
        adversarial_loss = torch.nn.BCELoss()
        for epoch in range(self.num_epoch):
            d_losses = 0
            g_losses = 0
            loop = tqdm(data_loader, total=len(data_loader))
            for datas in loop:
                valid = Variable(Tensor(datas.shape[0], 1).fill_(1.0),
                                 requires_grad=False)
                fake = Variable(Tensor(datas.shape[0], 1).fill_(0.0),
                                requires_grad=False)
                real_datas = Variable(datas.type(Tensor))
                optimizer_G.zero_grad()
                z = Variable(
                    Tensor(
                        np.random.normal(0, 1,
                                         (datas.shape[0], self.latent_dim))))
                gen_datas = netG(z)
                g_loss = adversarial_loss(netD(gen_datas), valid)
                g_loss.backward()
                optimizer_G.step()
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(netD(real_datas), valid)
                fake_loss = adversarial_loss(netD(gen_datas.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()
                loop.set_description(f'Epoch[{epoch}/{self.num_epoch}]')
                loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
                d_losses += d_loss.item()
                g_losses += g_loss.item()
        return netG, netD

    def generate_data(self, netG):
        if self.cuda:
            netG = netG.cuda()
        netG.eval()
        with torch.no_grad():
            z = torch.randn(100000, self.latent_dim)
            if self.cuda:
                fake_data = netG.forward(z.cuda())
            else:
                fake_data = netG.forward(z)
            fake_data = fake_data.cpu().numpy()
        return fake_data

    def fit(self, original_data):
        net_G, net_D = self.train(original_data)
        gen_data = self.generate_data(net_G)
        np.random.shuffle(gen_data)
        gen_data = gen_data[:self.num_gen]
        list_net = [net_G, net_D]
        return list_net, gen_data


class VAE:

    def __init__(self, num_gen, num_epoch, lr, batch_size, latent_dim):
        self.num_gen = num_gen
        self.num_epoch = num_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.cuda = True if cuda.is_available() else False
        self.setup_seed(42)

    def setup_seed(self, seed):
        manual_seed(seed)
        cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def loss_fcn(self, y, y_hat, mean, var):
        recon_loss = nn.BCELoss(reduction='sum')(y_hat, y)
        kl_loss = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var))
        return recon_loss + kl_loss * 0.001

    def train(self, train_data):
        if self.cuda:
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        data_dim = train_data.shape[1]
        if self.cuda:
            net_vae = model.Vae(data_dim=data_dim,
                                latent_dim=self.latent_dim).cuda()
        else:
            net_vae = model.Vae(data_dim=data_dim, latent_dim=self.latent_dim)
        dataset = torch.tensor(train_data)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        optimizer_vae = torch.optim.Adam(net_vae.parameters(), lr=self.lr)
        for epoch in range(self.num_epoch):
            loop = tqdm(data_loader, total=len(data_loader))
            for datas in loop:
                datas = Variable(datas.type(Tensor))
                if self.cuda:
                    datas = datas.to("cuda")
                recon, mean, var = net_vae(datas)
                recon_loss = nn.BCELoss(reduction='sum')(recon, datas)
                kl_loss = -0.5 * torch.sum(1 + var - mean**2 - torch.exp(var))
                loss = recon_loss + 0.0005 * kl_loss
                optimizer_vae.zero_grad()
                loss.backward()
                optimizer_vae.step()
                loop.set_description(f'Epoch[{epoch}/{self.num_epoch}]')
                loop.set_postfix(loss=loss.item(),
                                 recon_loss=recon_loss.item(),
                                 kl_loss=kl_loss.item())
        return net_vae

    def generate_data(self, vae):
        if self.cuda:
            vae = vae.cuda()
        vae.eval()
        with torch.no_grad():
            z = torch.randn(100000, self.latent_dim)
            if self.cuda:
                fake_data = vae.decoder(z.cuda())
            else:
                fake_data = vae.decoder(z)
            fake_data = fake_data.cpu().numpy()
        return fake_data

    def fit(self, original_data):
        net_vae = self.train(original_data)
        gen_data = self.generate_data(net_vae)
        np.random.shuffle(gen_data)
        gen_data = gen_data[:self.num_gen]
        list_net = [net_vae]
        return list_net, gen_data


class WGAN_GP:

    def __init__(self, num_gen, num_iters_g, lr, batch_size, latent_dim,
                 n_critic, Lambda):
        self.num_gen = num_gen
        self.num_iters_g = num_iters_g
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.Lambda = Lambda
        self.cuda = True if cuda.is_available() else False
        self.setup_seed(42)

    def setup_seed(self, seed):
        manual_seed(seed)
        cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def cal_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() if self.cuda else alpha
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        if self.cuda:
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        interpolates_D = netD(interpolates)
        gradients = torch.autograd.grad(
            outputs=interpolates_D,
            inputs=interpolates,
            grad_outputs=torch.ones(interpolates_D.size()).cuda()
            if self.cuda else torch.ones(interpolates_D.size()),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradients_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * self.Lambda
        return gradients_penalty

    def inf_train_data(self, train_data):
        dataset = torch.tensor(train_data)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        while True:
            for data in data_loader:
                yield data

    def train(self, train_data):
        if self.cuda:
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        data_dim = train_data.shape[1]
        if self.cuda:
            netG = model.WGan_generator(latent_dim=self.latent_dim,
                                        data_dim=data_dim).cuda()
            netD = model.WGan_discriminator(data_dim=data_dim).cuda()
        else:
            netG = model.WGan_generator(latent_dim=self.latent_dim,
                                        data_dim=data_dim)
            netD = model.WGan_discriminator(data_dim=data_dim)

        optimizer_G = torch.optim.Adam(netG.parameters(),
                                       lr=self.lr,
                                       betas=(0.5, 0.9))
        optimizer_D = torch.optim.Adam(netD.parameters(),
                                       lr=self.lr,
                                       betas=(0.5, 0.9))
        datas = self.inf_train_data(train_data=train_data)
        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()
        for iter_g in range(self.num_iters_g):
            '''update D network'''
            for p in netD.parameters():
                p.requires_grad = True
            for iter_d in range(self.n_critic):
                data = next(datas)
                real_data = Variable(data.type(Tensor))
                netD.zero_grad()
                # train with real
                D_real = netD(real_data)
                D_real = D_real.mean()
                # train with fake
                noise = torch.randn(real_data.size(0), self.latent_dim)
                noise = Variable(noise.type(Tensor), volatile=True)
                fake = Variable(netG(noise).data)
                D_fake = netD(fake).mean()
                # train with gp
                gradient_penalty = self.cal_gradient_penalty(
                    netD, real_data.data, fake.data)
                D_loss = D_fake - D_real + gradient_penalty
                D_loss.backward()
                Wasserstein_D = D_real - D_fake
                optimizer_D.step()
            '''update G network'''
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            nosie_g = torch.randn(self.batch_size, self.latent_dim)
            nosie_g = Variable(nosie_g.type(Tensor))
            fake = netG(nosie_g)
            d_fake = netD(fake).mean()
            G_loss = -d_fake
            G_loss.backward()
            optimizer_G.step()
            if iter_g % 100 == 0:
                print('iter=', iter_g, ': D_loss=', D_loss, ', G_cost=',
                      G_loss, ': W_D=', Wasserstein_D)
        return netG, netD

    def generate_data(self, netG):
        if self.cuda:
            netG = netG.cuda()
        netG.eval()
        with torch.no_grad():
            z = torch.randn(100000, self.latent_dim)
            if self.cuda:
                fake_data = netG.forward(z.cuda())
            else:
                fake_data = netG.forward(z)
            fake_data = fake_data.cpu().numpy()
        return fake_data

    def fit(self, original_data):
        net_G, net_D = self.train(original_data)
        gen_data = self.generate_data(net_G)
        np.random.shuffle(gen_data)
        gen_data = gen_data[:self.num_gen]
        list_net = [net_G, net_D]
        return list_net, gen_data
    
    
class LSGAN:
    def __init__(self, num_gen, num_iters_g, lr, batch_size, latent_dim,
                 n_critic):
        self.num_gen = num_gen
        self.num_iters_g = num_iters_g
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.cuda = True if cuda.is_available() else False
        self.setup_seed(42)
    
    def setup_seed(self, seed):
        manual_seed(seed)
        cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def inf_train_data(self, train_data):
        dataset = torch.tensor(train_data)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        while True:
            for data in data_loader:
                yield data
    
    def train(self, train_data):
        if self.cuda:
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        data_dim = train_data.shape[1]
        if self.cuda:
            netG = model.WGan_generator(latent_dim=self.latent_dim,
                                        data_dim=data_dim).cuda()
            netD = model.WGan_discriminator(data_dim=data_dim).cuda()
        else:
            netG = model.WGan_generator(latent_dim=self.latent_dim,
                                        data_dim=data_dim)
            netD = model.WGan_discriminator(data_dim=data_dim)

        optimizer_G = torch.optim.Adam(netG.parameters(),
                                       lr=self.lr,
                                       betas=(0.5, 0.9))
        optimizer_D = torch.optim.Adam(netD.parameters(),
                                       lr=self.lr,
                                       betas=(0.5, 0.9))
        datas = self.inf_train_data(train_data=train_data)
        for iter_g in range(self.num_iters_g):
            '''update D network'''
            for p in netD.parameters():
                p.requires_grad = True
            for iter_d in range(self.n_critic):
                data = next(datas)
                real_data = Variable(data.type(Tensor))
                netD.zero_grad()
                D_real = netD(real_data)
                noise = torch.randn(real_data.size(0), self.latent_dim)
                noise = Variable(noise.type(Tensor), volatile=True)
                fake = Variable(netG(noise).data)
                D_fake = netD(fake)
                D_loss = 0.5*(torch.mean((D_real-1)**2) + torch.mean((D_fake)**2))
                D_loss.backward()
                optimizer_D.step()
            '''update G network'''
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            nosie_g = torch.randn(self.batch_size, self.latent_dim)
            nosie_g = Variable(nosie_g.type(Tensor))
            fake = netG(nosie_g)
            d_fake = netD(fake)
            G_loss = 0.5 * torch.mean((d_fake - 1) ** 2)
            G_loss.backward()
            optimizer_G.step()
            if iter_g % 100 == 0:
                print('iter=', iter_g, ': D_loss=', D_loss, ', G_cost=',
                      G_loss)
        return netG, netD
    
    def generate_data(self, netG):
        if self.cuda:
            netG = netG.cuda()
        netG.eval()
        with torch.no_grad():
            z = torch.randn(100000, self.latent_dim)
            if self.cuda:
                fake_data = netG.forward(z.cuda())
            else:
                fake_data = netG.forward(z)
            fake_data = fake_data.cpu().numpy()
        return fake_data

    def fit(self, original_data):
        net_G, net_D = self.train(original_data)
        gen_data = self.generate_data(net_G)
        np.random.shuffle(gen_data)
        gen_data = gen_data[:self.num_gen]
        list_net = [net_G, net_D]
        return list_net, gen_data
    

class DDPM:
    def __init__(self, num_gen, num_epochs, lr, batch_size, num_steps):
        self.num_gen = num_gen
        self.num_epoch = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_dim = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        setup_seed(42)
    
    def corrupt(self, x, amount):
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1)
        return x*(1-amount) + noise*amount 
    
    def train(self, train_data):
        self.data_dim = train_data.shape[1]
        dataset = torch.tensor(train_data)
        dataset = dataset.view(train_data.shape[0], 1, train_data.shape[1])
        train_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        net = model.BasicUnet().to(self.device)
        loss_fn = nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        losses = []
        for epoch in range(self.num_epoch):
            for x in train_dataloader:
                x = x.float().to(self.device) 
                noise_amount = torch.rand(x.shape[0]).float().to(self.device) 
                noisy_x = self.corrupt(x, noise_amount) 
                pred = net(noisy_x)
                loss = loss_fn(pred, x) 
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
            print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
            # loop.set_description(f'Epoch[{epoch}/{self.num_epoch}]')
            # loop.set_postfix(loss=avg_loss)
        return net
    
    def generate_data(self, net):
        net = net.to(self.device)
        x = torch.rand(self.num_gen, 1, self.data_dim).to(self.device)
        for i in range(self.num_steps):
            with torch.no_grad():
                pred = net(x)
            mix_factor = 1/(self.num_steps - i)
            x = x*(1-mix_factor) + pred*mix_factor
        x = x.view(self.num_gen, self.data_dim)
        gen_data = np.array(x.detach().cpu())
        return gen_data
    
    def fit(self, original_data):
        net_ddpm = self.train(original_data)
        gen_data = self.generate_data(net_ddpm)
        return [net_ddpm], gen_data


class VAEGAN:
    def __init__(self, num_gen, num_epoch, lr, batch_size, latent_dim):
        self.num_gen = num_gen
        self.num_epoch = num_epoch
        self.lr = lr
        self.data_dim = 0
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        setup_seed(42)
    
    def lossD(self, scores_real, scores_fake0, scores_fake1):
        loss = torch.sum(-torch.log(scores_real)) + torch.sum(-torch.log(1 - scores_fake0)) + torch.sum(-torch.log(1 - scores_fake1))
        return loss

    def train(self, train_data):
        dataset = torch.tensor(train_data)
        self.data_dim = train_data.shape[1]
        dataset = dataset.view(train_data.shape[0], 1, train_data.shape[1])
        train_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        net_decoder = model.Vaegan_Decoder(data_dim=train_data.shape[1], z_size=self.latent_dim).to(self.device)
        net_encoder = model.Vaegan_Encoder(data_dim=train_data.shape[1], z_size=self.latent_dim).to(self.device)
        net_discriminator = model.Vaegan_Discriminator(data_dim=train_data.shape[1]).to(self.device)
        optimizer_encoder = torch.optim.RMSprop(params=net_encoder.parameters(),lr=self.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
        lr_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_encoder, gamma=0.9)
        optimizer_decoder = torch.optim.RMSprop(params=net_decoder.parameters(),lr=self.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
        lr_decoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_decoder, gamma=0.9)
        optimizer_discriminator = torch.optim.RMSprop(params=net_discriminator.parameters(),lr=self.lr,alpha=0.9,eps=1e-8,weight_decay=0,momentum=0,centered=False)
        lr_discriminator = torch.optim.lr_scheduler.ExponentialLR(optimizer_discriminator, gamma=0.9)
        for epoch in range(self.num_epoch):
            net_encoder.train()
            net_discriminator.train()
            net_decoder.train()
            loss_en = []
            loss_de = []
            loss_di = []
            for x in train_dataloader:
                
                '''train decoder and encoder'''
                x = Variable(x, requires_grad=True).float().to(self.device)
                z, kl = net_encoder(x)
                x_z = net_decoder(z)
                z_p = Variable(torch.randn(len(x), self.latent_dim), requires_grad=True).to(self.device)
                x_p = net_decoder(z_p)
                d_x = net_discriminator(x, 'GAN')
                d_x_z = net_discriminator(x_z, 'GAN')
                d_x_p = net_discriminator(x_p, 'GAN')
                f_x = net_discriminator(x, 'REC')
                f_x_z = net_discriminator(x_z, 'REC')
                f_x_p = net_discriminator(x_p, 'REC')
                loss_encoder = torch.sum(kl)+torch.sum(torch.sum(0.5*(f_x - f_x_z) ** 2, 1))
                loss_decoder = 0.01 * torch.sum(torch.sum(0.5*(f_x - f_x_z) ** 2, 1)) - self.lossD(d_x, d_x_z, d_x_p)
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                loss_encoder.backward(retain_graph=True)
                loss_decoder.backward(retain_graph=True)
                optimizer_encoder.step()
                optimizer_decoder.step()
                '''train discriminator'''
                x = Variable(x, requires_grad=True).float().to(self.device)
                z, kl = net_encoder(x)
                x_z = net_decoder(z)
                z_p = Variable(torch.randn(len(x), self.latent_dim), requires_grad=True).to(self.device)
                x_p = net_decoder(z_p)
                d_x = net_discriminator(x, 'GAN')
                d_x_z = net_discriminator(x_z, 'GAN')
                d_x_p = net_discriminator(x_p, 'GAN')
                loss_discriminator = self.lossD(d_x, d_x_z, d_x_p)
                optimizer_discriminator.zero_grad()
                loss_discriminator.backward()
                optimizer_discriminator.step()
                loss_en.append(loss_encoder)
                loss_de.append(loss_decoder)
                loss_di.append(loss_discriminator)
            print('[%02d] encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (epoch, sum(loss_en)/len(loss_en), sum(loss_de)/len(loss_de), sum(loss_di)/len(loss_di)))

            lr_encoder.step()
            lr_decoder.step()
            lr_discriminator.step()
        return net_encoder, net_decoder, net_discriminator

    def generate_data(self, net):
        net = net.to(self.device)
        net.eval()
        z = torch.randn(self.num_gen, self.latent_dim).to(self.device)
        out = net(z).view(self.num_gen, self.data_dim)
        out = out.cpu().detach().numpy()
        return out
    
    def fit(self, original_data):
        net_encoder, net_decoder, net_discriminator = self.train(original_data)
        gen_data = self.generate_data(net_decoder)
        list_net = [net_encoder, net_decoder, net_discriminator]
        return list_net, gen_data
    

class FLOW:
    def __init__(self, num_gen, num_epochs, lr, batch_size, num_blocks):
        self.num_gen = num_gen
        self.num_epoch = num_epochs
        self.lr = lr
        self.data_dim = 0
        self.batch_size = batch_size
        self.num_blocks = num_blocks
        self.num_hidden = 64
        self.model = None
        self.optimizer = None
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        setup_seed(42)

    def train(self, train_data):
        train_tensor = torch.tensor(train_data).float()
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        for epoch in range(self.num_epoch):
            print('\nEpoch: {}'.format(epoch))
            self.model.train()
            train_loss = 0
            pbar = tqdm(total=len(train_loader.dataset))
            for batch_idx, data in enumerate(train_loader):
                if isinstance(data, list):
                    data = data[0]
                data = data.float().to(self.device)
                self.optimizer.zero_grad()
                loss = -self.model.log_probs(data).mean()
                train_loss = train_loss + loss.item()
                loss.backward()
                self.optimizer.step()
                pbar.update(data.size(0))
                pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(-train_loss/(batch_idx + 1)))
            pbar.close()
            for module in self.model.modules():
                if isinstance(module, model.BatchNormFlow):
                    module.momentum = 0
            with torch.no_grad():
                self.model(train_loader.dataset.tensors[0].to(self.device))
            for module in self.model.modules():
                if isinstance(module, model.BatchNormFlow):
                    module.momentum = 1
    
    def outlier_cull(self, gen_data):
        gen_data = np.delete(gen_data, np.where(np.isnan(gen_data))[0], axis=0)
        mean = np.mean(gen_data, axis=0)
        cov = np.cov(gen_data, rowvar=False)
        distances = [mahalanobis(gen_data[i], mean, np.linalg.inv(cov)) for i in range(len(gen_data))]
        outliers = distances > np.mean(distances)+3*np.std(distances)
        gen_data_no_outlier = gen_data[outliers == 0]
        return gen_data_no_outlier
    
    def generate_data(self):
        self.model.eval()
        noise = torch.Tensor(self.num_gen * 2, self.data_dim).normal_()
        with torch.no_grad():
            gen_data = self.model.sample(self.num_gen * 2, noise).detach().cpu().numpy()
        gen_data_no_outlier = self.outlier_cull(gen_data)
        np.random.shuffle(gen_data_no_outlier)
        gen_data_final = gen_data_no_outlier[:self.num_gen]
        return gen_data_final


class MAF(FLOW):
    '''MASKED AUTOREGRESSIVE FLOW'''
    def __init__(self, num_gen, num_epochs, lr, batch_size, num_blocks):
        super().__init__(num_gen, num_epochs, lr, batch_size, num_blocks)
    
    def fit(self, original_data):
        self.data_dim = original_data.shape[1]
        modules = []
        for _ in range(self.num_blocks):
            modules += [model.MADE(num_inputs=self.data_dim, num_hidden=self.num_hidden, num_cond_inputs=None, act='relu'),
                        model.BatchNormFlow(num_inputs=self.data_dim),
                        model.Reverse(num_inputs=self.data_dim)]
        self.model = model.FlowSequential(*modules)
        self.train(train_data=original_data)
        gen_data = self.generate_data()
        return [self.model], gen_data


class REALNVP(FLOW):
    '''Real-valued Non-Volume Preserving'''
    def __init__(self, num_gen, num_epochs, lr, batch_size, num_blocks):
        super().__init__(num_gen, num_epochs, lr, batch_size, num_blocks)
    
    def fit(self, original_data):
        self.data_dim = original_data.shape[1]
        modules = []
        mask = torch.arange(0, self.data_dim) % 2
        mask = mask.to(self.device).float()
        for _ in range(self.num_blocks):
            modules += [model.CouplingLayer(num_inputs=self.data_dim, num_hidden=self.num_hidden, mask=mask, num_cond_inputs=None,
                                            s_act='tanh', t_act='relu'),
                        model.BatchNormFlow(num_inputs=self.data_dim)]
            mask = 1 - mask
        self.model = model.FlowSequential(*modules)
        self.train(train_data=original_data)
        gen_data = self.generate_data()
        return [self.model], gen_data


class GLOW(FLOW):
    '''GLOW'''
    def __init__(self, num_gen, num_epochs, lr, batch_size, num_blocks):
        super().__init__(num_gen, num_epochs, lr, batch_size, num_blocks)
    
    def fit(self, original_data):
        self.data_dim = original_data.shape[1]
        modules = []
        mask = torch.arange(0, self.data_dim) % 2
        mask = mask.to(self.device).float()
        for _ in range(self.num_blocks):
            modules += [model.BatchNormFlow(num_inputs=self.data_dim),
                        model.LUInvertibleMM(num_inputs=self.data_dim),
                        model.CouplingLayer(num_inputs=self.data_dim, num_hidden=self.num_hidden, mask=mask, num_cond_inputs=None,
                                            s_act='tanh', t_act='relu'),
                        ]
            mask = 1 - mask
        self.model = model.FlowSequential(*modules)
        self.train(train_data=original_data)
        gen_data = self.generate_data()
        return [self.model], gen_data





    
    





    

        







        
        



    


    

# class DDPM:
#     def __init__(self, num_gen, num_epochs, lr, batch_size, num_steps):
#         self.num_gen = num_gen
#         self.num_epochs = num_epochs
#         self.lr = lr
#         self.batch_size = batch_size
#         self.num_steps = num_steps
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.setup_seed(42)
#         betas = torch.linspace(-6, 6, num_steps)
#         self.betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
#         self.alphas = 1 - self.betas
#         self.alphas_prod = torch.cumprod(self.alphas, 0)
#         self.alphas_prod_p = torch.cat(
#             [torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
#         self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
#         self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
#         self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

#     def setup_seed(self, seed):
#         manual_seed(seed)
#         cuda.manual_seed_all(seed)
#         np.random.seed(seed)
#         torch.backends.cudnn.deterministic = True

#     def diffusion_loss_fn(self, model, x_0, alphas_bar_sqrt,
#                           one_minus_alphas_bar_sqrt, n_steps):
#         batch_size = x_0.shape[0]
#         t = torch.randint(0, n_steps, size=(batch_size // 2, ))
#         t = torch.cat([t, n_steps - 1 - t], dim=0)
#         t = t.unsqueeze(-1)
#         t = t.to(device=self.device)
#         a = alphas_bar_sqrt[t]
#         aml = one_minus_alphas_bar_sqrt[t]
#         e = torch.randn_like(x_0).to(self.device)
#         x = x_0.to(self.device) * a.to(self.device) + e.to(
#             self.device) * aml.to(self.device)
#         x = x.to(self.device)
#         output = model(x, t.squeeze(-1)).to(self.device)
#         return (e - output).square().mean()

#     def p_sample_loop(self, model, shape, n_steps, betas,
#                       one_minus_alphas_bar_sqrt):
#         cur_x = torch.randn(self.num_gen, shape).to(self.device)
#         x_seq = [cur_x]
#         for i in reversed(range(n_steps)):
#             cur_x = self.p_sample(model, cur_x, i, betas,
#                                   one_minus_alphas_bar_sqrt)
#             x_seq.append(cur_x)
#         return x_seq

#     def p_sample(self, model, x, t, betas, one_minus_alphas_bar_sqrt):
#         t = torch.tensor([t]).to(self.device)
#         x = x.to(self.device)
#         coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
#         eps_theta = model(x, t)
#         mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
#         z = torch.randn_like(x).to(self.device)
#         sigma_t = betas[t].sqrt().to(self.device)
#         sample = mean + sigma_t * z
#         return (sample)

#     def train(self, train_data):
#         self.data_dim = train_data.shape[1]
#         dataset = torch.Tensor(train_data).float()
#         data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                   batch_size=self.batch_size,
#                                                   shuffle=True)
#         net_ddpm = model.MLPDiffusion(n_steps=self.num_steps,
#                                       data_dim=self.data_dim).to(self.device)
#         optimizer_ddpm = torch.optim.Adam(net_ddpm.parameters(), lr=self.lr)
#         for epoch in range(self.num_epochs):
#             loop = tqdm(data_loader, total=len(data_loader))
#             for datas in loop:
#                 # datas = Variable(datas.type(Tensor)).to(self.device)
#                 loss = self.diffusion_loss_fn(net_ddpm, datas,
#                                               self.alphas_bar_sqrt,
#                                               self.one_minus_alphas_bar_sqrt,
#                                               self.num_steps).to(self.device)
#                 optimizer_ddpm.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(net_ddpm.parameters(), 1.)
#                 optimizer_ddpm.step()
#                 loop.set_description(f'Epoch[{epoch}/{self.num_epochs}]')
#                 loop.set_postfix(loss=loss.item())
#         return net_ddpm

#     def generate_data(self, model_ddpm):
#         model_ddpm = model_ddpm.to(self.device)
#         model_ddpm.eval()
#         x_seq = self.p_sample_loop(model_ddpm, self.data_dim, self.num_steps,
#                                    self.betas, self.one_minus_alphas_bar_sqrt)
#         gen_data = x_seq[10 * 10].detach().cpu().numpy()
#         return gen_data

#     def fit(self, original_data):
#         net_ddpm = self.train(original_data)
#         gen_data = self.generate_data(net_ddpm)
#         torch.save(net_ddpm.state_dict(), 'diffusion_soft.pkl')
#         np.savetxt('gen_data.txt', gen_data)
#         list_net = [net_ddpm]
#         return list_net, gen_data

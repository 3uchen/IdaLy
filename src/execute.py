import augmentation as aug


def GMM_execute(original_data, num_gen, para):
    gmm = aug.Gmm(N=num_gen, n_components=para[0])
    gmm_gen = gmm.fit(original_data)
    return gmm_gen


def GAN_execute(original_data, num_gen, para):
    gan = aug.GAN(num_gen=num_gen, num_epoch=para[0], lr=para[1],
                  batch_size=para[2], latent_dim=para[3])
    list_net, gen_data = gan.fit(original_data)
    return list_net, gen_data


def WGAN_GP_execute(original_data, num_gen, para):
    wgan_gp = aug.WGAN_GP(num_gen=num_gen, num_iters_g=para[0], lr=para[1],
                          batch_size=para[2], latent_dim=para[3],
                          n_critic=para[4], Lambda=para[5])
    list_net, gen_data = wgan_gp.fit(original_data)
    return list_net, gen_data


def LSGAN_execute(original_data, num_gen, para):
    lsgan = aug.LSGAN(num_gen=num_gen, num_iters_g=para[0], lr=para[1],
                      batch_size=para[2], latent_dim=para[3],
                      n_critic=para[4])
    list_net, gen_data = lsgan.fit(original_data)
    return list_net, gen_data


def VAE_execute(original_data, num_gen, para):
    vae = aug.VAE(num_gen=num_gen, num_epoch=para[0], lr=para[1],
                  batch_size=para[2], latent_dim=para[3])
    list_net, gen_data = vae.fit(original_data)
    return list_net, gen_data


def VAEGAN_execute(original_data, num_gen, para):
    vaegan = aug.VAEGAN(num_gen=num_gen, num_epoch=para[0], lr=para[1],
                        batch_size=para[2], latent_dim=para[3])
    list_net, gen_data = vaegan.fit(original_data)
    return list_net, gen_data


def DDPM_execute(original_data, num_gen, para):
    ddpm = aug.DDPM(num_gen=num_gen, num_epochs=para[0], lr=para[1],
                    batch_size=para[2], num_steps=para[3])
    list_net, gen_data = ddpm.fit(original_data)
    return list_net, gen_data


def MAF_execute(original_data, num_gen, para):
    maf = aug.MAF(num_gen=num_gen, num_epochs=para[0], lr=para[1],
                  batch_size=para[2], num_blocks=para[3])
    list_net, gen_data = maf.fit(original_data)
    return list_net, gen_data


def REALNVP_execute(original_data, num_gen, para):
    realnvp = aug.REALNVP(num_gen=num_gen, num_epochs=para[0], lr=para[1],
                          batch_size=para[2], num_blocks=para[3])
    list_net, gen_data = realnvp.fit(original_data)
    return list_net, gen_data


def GLOW_execute(original_data, num_gen, para):
    glow = aug.GLOW(num_gen=num_gen, num_epochs=para[0], lr=para[1],
                    batch_size=para[2], num_blocks=para[3])
    list_net, gen_data = glow.fit(original_data)
    return list_net, gen_data


def GNI_execute(original_data, num_gen, para):
    gni = aug.GNI(num_gen=num_gen, mean=para[0], variance=para[1])
    gni_gen = gni.fit(original_data)
    return gni_gen


def SNI_execute(original_data, num_gen, para):
    sni = aug.SNI(num_gen=num_gen, mean=para[0], variance=para[1])
    sni_gen = sni.fit(original_data)
    return sni_gen

def Cutout_execute(original_data, num_gen, para):
    cutout = aug.Cutout(num_gen=num_gen, mask_prob=para[0])
    cutout_gen = cutout.fit(original_data)
    return cutout_gen


def PNI_execute(original_data, num_gen, para):
    pni = aug.PNI(num_gen=num_gen)
    pni_gen = pni.fit(original_data)
    return pni_gen


def MSI_execute(original_data, num_gen, para):
    msi = aug.MSI(num_gen=num_gen, mask_prob=para[0])
    msi_gen = msi.fit(original_data)
    return msi_gen


def KNNMTD_execute(original_data, num_gen, para):
    knnMTD = aug.kNNMTD(n_obs=num_gen, k=para[0])
    knnMTD_gen = knnMTD.fit(original_data)
    return knnMTD_gen


def LLE_execute(original_data, num_gen, para):
    lle = aug.Lle(num_gen=num_gen, n_neighbor=para[0], reg=para[1],
                  n_component=para[2])
    lle_gen = lle.fit(original_data)
    return lle_gen


def MTD_execute(original_data, num_gen, para):
    mtd = aug.MTD(n_obs=num_gen)
    MTD_gen = mtd.fit(original_data)
    return MTD_gen


def MIXUP_execute(original_data, num_gen, para):
    mixup = aug.Mixup(num_gen=num_gen, alpha=para[0])
    mixup_gen = mixup.fit(original_data)
    return mixup_gen


def SMOTE_execute(original_data, num_gen, para):
    smote = aug.Smote(N=num_gen, k=para[0], r=2)
    smote_gen = smote.fit(original_data)
    return smote_gen


def Kmeans_SMOTE_execute(original_data, num_gen, para):
    kmeans_smote = aug.Kmeans_Smote(num_gen=num_gen, n_clusters=para[0])
    kmeans_smote_gen = kmeans_smote.fit(original_data)
    return kmeans_smote_gen



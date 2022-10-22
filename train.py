#import argparse
from fasttext import FastText
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import VisualSemanticEmbedding
from model import Generator, Discriminator
from data import ReedICML2016
import math




img_root = './datasets'
caption_root = './datasets/FashionGAN_txt'
#trainclasses_file = './datasets/FashionGAN_txt/trainclasses.txt'
trainclasses_file = 'trainclasses.txt'
fasttext_model = './fasttext_models/wiki.en.bin'
text_embedding_model = './models/text_embedding_fashion.pth'
save_filename = './models/fashion_res_lowrank_64.pth'
num_threads = 4
num_epochs = 100
batch_size = 64
learning_rate = 0.0002
lr_decay = 0.5
momentum = 0.5
embed_ndim = 300
max_nwords = 50
use_vgg = True
no_cuda = False
fusing_method = 'lowrank_BP'



if not no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    no_cuda = True


def preprocess(img, desc, len_desc, txt_encoder):
    # img = Variable(img.cuda() if not no_cuda else img)
    desc = Variable(desc)

    len_desc = len_desc.numpy()
    sorted_indices = np.argsort(len_desc)[::-1]
    original_indices = np.argsort(sorted_indices)
    packed_desc = nn.utils.rnn.pack_padded_sequence(
        desc[torch.LongTensor(sorted_indices.copy()), ...].transpose(0, 1),
        len_desc[sorted_indices]
    )
    _, txt_feat = txt_encoder(packed_desc)
    txt_feat = txt_feat.squeeze()
    txt_feat = txt_feat[original_indices, ...]

    txt_feat_np = txt_feat.data.numpy()
    txt_feat_mismatch = torch.Tensor(np.roll(txt_feat_np, 1, axis=0))
    txt_feat_mismatch = Variable(txt_feat_mismatch)
    txt_feat_np_split = np.split(txt_feat_np, [txt_feat_np.shape[0] // 2])
    txt_feat_relevant = torch.Tensor(np.concatenate([
        np.roll(txt_feat_np_split[0], -1, axis=0),
        txt_feat_np_split[1]
    ]))
    txt_feat_relevant = Variable(txt_feat_relevant)
    return img, txt_feat, txt_feat_mismatch, txt_feat_relevant


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = FastText.load_model(fasttext_model)

    print('Loading a dataset...')
    train_data = ReedICML2016(img_root,
                              caption_root,
                              trainclasses_file,
                              word_embedding,
                              max_nwords,
                              transforms.Compose([
                                  transforms.Scale(74),
                                  transforms.RandomCrop(64),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()
                              ]))
    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_threads)

    word_embedding = None

    # pretrained text embedding model
    print('Loading a pretrained text embedding model...')
    txt_encoder = VisualSemanticEmbedding(embed_ndim)
    txt_encoder.load_state_dict(torch.load(text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder
    for param in txt_encoder.parameters():
        param.requires_grad = False

    G = Generator(use_vgg=use_vgg, fusing=fusing_method)
    D = Discriminator(fusing_method=fusing_method)

    if not no_cuda:
        txt_encoder 
        G 
        D 

    g_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, G.parameters()),
                                   lr=learning_rate, betas=(momentum, 0.999))
    d_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, D.parameters()),
                                   lr=learning_rate, betas=(momentum, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, lr_decay)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, lr_decay)

    for epoch in range(num_epochs):
        d_lr_scheduler.step()
        g_lr_scheduler.step()

        # training loop
        avg_D_real_loss = 0
        avg_D_real_m_loss = 0
        avg_D_fake_loss = 0
        avg_G_fake_loss = 0
        avg_kld = 0
        for i, (img, desc, len_desc) in enumerate(train_loader):
            img, txt_feat, txt_feat_mismatch, txt_feat_relevant = \
                preprocess(img, desc, len_desc, txt_encoder)
            img_norm = img * 2 - 1
            vgg_norm1 = (img[:,0,:,:] - 0.485)/0.229
            vgg_norm2 = (img[:, 1, :, :] - 0.456) / 0.224
            vgg_norm3 = (img[:, 2, :, :] - 0.406) / 0.225
            vgg_norm = torch.cat((vgg_norm1.unsqueeze(1),vgg_norm2.unsqueeze(1),vgg_norm3.unsqueeze(1)), 1)
            img_norm = Variable(img_norm)
            img_G = Variable(vgg_norm) if use_vgg else img_norm

            ONES = Variable(torch.ones(img.size(0)))
            ZEROS = Variable(torch.zeros(img.size(0)))
            if not no_cuda:
                ONES, ZEROS = ONES, ZEROS

            if i % 2 == 0:
                # UPDATE DISCRIMINATOR
                D.zero_grad()
                # real image with matching text
                real_logit = D(img_norm, txt_feat)
                #real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
                real_loss = F.mse_loss(F.sigmoid(real_logit), ONES)
                avg_D_real_loss += real_loss.data 
                real_loss.backward()
                # real image with mismatching text
                real_m_logit = D(img_norm, txt_feat_mismatch)
                #real_m_loss = 0.5 * F.binary_cross_entropy_with_logits(real_m_logit, ZEROS)
                real_m_loss = 0.5 * F.mse_loss(F.sigmoid(real_m_logit), ZEROS)
                avg_D_real_m_loss += real_m_loss.data 
                real_m_loss.backward()
                # synthesized image with semantically relevant text
                fake, _ = G(img_G, txt_feat_relevant)
                fake_logit = D(fake.detach(), txt_feat_relevant)
                fake_loss = 0.5 * F.mse_loss(F.sigmoid(fake_logit), ZEROS)
                avg_D_fake_loss += fake_loss.data 
                fake_loss.backward()
                d_optimizer.step()

            # UPDATE GENERATOR
            G.zero_grad()
            fake, (z_mean, z_log_stddev) = G(img_G, txt_feat_relevant)
            kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1))
            avg_kld += kld.data 
            x = 0
            x += kld.data
            fake_logit = D(fake, txt_feat_relevant)
            fake_loss = F.mse_loss(F.sigmoid(fake_logit), ONES)
            avg_G_fake_loss += fake_loss.data 
            G_loss = fake_loss + kld
            G_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], D_real: %.4f, D_mis: %.4f, D_fake: %.4f, G_fake: %.4f, KLD: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader), avg_D_real_loss / (i + 1),
                      avg_D_real_m_loss / (i + 1), avg_D_fake_loss / (i + 1), avg_G_fake_loss / (i + 1), avg_kld / (i + 1)))

        save_image((fake.data + 1) * 0.5, './examples/epoch_%d.png' % (epoch + 1))
        if epoch % 10 ==0:
            torch.save(G.state_dict(), save_filename)    
    avg = x / (i + 1) 
    is_score = math.exp(avg)
    print("is_score=", is_score)
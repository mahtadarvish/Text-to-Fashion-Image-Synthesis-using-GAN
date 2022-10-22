#import argparse
from fasttext import FastText
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import VisualSemanticEmbedding
from data import ReedICML2016
#Path = "C:/Users/mansouri/Downloads/Mahta Darvish/BilinearGAN_for_LBIE-master/datasets/FashionGAN_txt/trainclasses.txt"

 
img_root = './datasets'
caption_root = './datasets/FashionGAN_txt'
#trainclasses_file = './datasets/FashionGAN_txt/trainclasses.txt'
trainclasses_file = 'trainclasses.txt'
fasttext_model = './fasttext_models/wiki.en.bin'
save_filename = './models/text_embedding_fashion.pth'
num_threads = 4
num_epochs = 5
batch_size = 64
learning_rate = 0.0002
margin = 0.2
embed_ndim = 300
max_nwords = 50
no_cuda = False


if not no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    no_cuda = True


def pairwise_ranking_loss(margin, x, v):
    zero = torch.zeros(1)
    diag_margin = margin * torch.eye(x.size(0))
    if not no_cuda:
        zero, diag_margin = zero.cuda(), diag_margin.cuda()
    zero, diag_margin = Variable(zero), Variable(diag_margin)

    x = x / torch.norm(x, 2, 1, keepdim=True)
    v = v / torch.norm(v, 2, 1, keepdim=True)
    prod = torch.matmul(x, v.transpose(0, 1))
    diag = torch.diag(prod)
    for_x = torch.max(zero, margin - torch.unsqueeze(diag, 1) + prod) - diag_margin
    for_v = torch.max(zero, margin - torch.unsqueeze(diag, 0) + prod) - diag_margin
    return (torch.sum(for_x) + torch.sum(for_v)) / x.size(0)


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = FastText.load_model('./fasttext_models/wiki.en.bin')

    print('Loading a dataset...')
    train_data = ReedICML2016(img_root,
                              caption_root,
                              trainclasses_file,
                              word_embedding,
                              max_nwords,
                              transforms.Compose([
                                  transforms.Scale(256),
                                  transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                              ]))

    word_embedding = None

    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_threads)

    model = VisualSemanticEmbedding(embed_ndim)
    if not no_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                 lr=learning_rate)

    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (img, desc, len_desc) in enumerate(train_loader):
            img = Variable(img.cuda() if not no_cuda else img)
            desc = Variable(desc.cuda() if not no_cuda else desc)
            len_desc, indices = torch.sort(len_desc, 0, True)
            indices = indices.numpy()
            img = img[indices, ...]
            desc = desc[indices, ...].transpose(0, 1)
            desc = nn.utils.rnn.pack_padded_sequence(desc, len_desc.numpy())

            optimizer.zero_grad()
            img_feat, txt_feat = model(img, desc)
            loss = pairwise_ranking_loss(margin, img_feat, txt_feat)
            avg_loss += loss.data  
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader), avg_loss / (i + 1)))

        torch.save(model.state_dict(), save_filename)

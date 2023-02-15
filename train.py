from __future__ import print_function
import argparse
import time
import setproctitle
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader import VideoDataset_train, VideoDataset_test
from data_manager import VCM
from eval_metrics import evaluate
from loss import OriTripletLoss, EasySampleLoss
from model_main import embed_net
from utils import *

setproctitle.setproctitle("刘明慧")

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='VCM', help='dataset name: VCM(Video Cross-modal)')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='best_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log_ddag/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')#！
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')#!
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')#64
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')#!
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='2', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')
parser.add_argument('--a', default=1, type=float,
                    metavar='lambda1', help='dropout ratio')


args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
dataset = args.dataset

#添加
seq_lenth = 6
test_batch = 32
data_set = VCM()
log_path = args.log_path + 'VCM_log/'
test_mode = [1, 2]
height = args.img_h
width = args.img_w

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

# log file name
suffix = dataset

suffix = suffix + '_drop_{}_{}_{}_lr_{}_seed_{}'.format(args.drop, args.num_pos, args.batch_size, args.lr, args.seed)
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_v2t = 0

best_map_acc = 0  # best test accuracy
best_map_acc_v2t = 0

start_epoch = 0
feature_dim = args.low_dim
wG = 0
end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,])


if dataset == 'VCM':
    rgb_pos, ir_pos = GenIdx(data_set.rgb_label, data_set.ir_label)

queryloader = DataLoader(
              VideoDataset_test(data_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test),
              batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader = DataLoader(
                VideoDataset_test(data_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test),
                batch_size=test_batch, shuffle=False, num_workers=args.workers)
# ----------------visible to infrared----------------
queryloader_1 = DataLoader(
                VideoDataset_test(data_set.query_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
                batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader_1 = DataLoader(
                  VideoDataset_test(data_set.gallery_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
                  batch_size=test_batch, shuffle=False, num_workers=args.workers)


n_class  = data_set.num_train_pids

nquery_1 = data_set.num_query_tracklets_1
ngall_1  = data_set.num_gallery_tracklets_1
nquery   = data_set.num_query_tracklets
ngall    = data_set.num_gallery_tracklets

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=args.drop, part=args.part, arch=args.arch, wpa=args.wpa)
net.to(device)

# checkpoint = load_checkpoint(osp.join( args.model_path, 'VCM_0.2_2_4_lr_0.1_seed_0t2v_rank1_best.t'))
# net.load_state_dict(checkpoint['net'])
# start_epoch = checkpoint['epoch']

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#
# define loss function
loader_batch = args.batch_size * args.num_pos
criterion1   = nn.CrossEntropyLoss()
criterion2   = OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_cg = nn.MSELoss()
criterion3   = EasySampleLoss(batch_size=loader_batch, margin=args.margin)

criterion1.to(device)
criterion2.to(device)
criterion_cg.to(device)
criterion3.to(device)

# optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_P = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer_P, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 35:
        lr = args.lr
    elif 35 <= epoch < 80:
        lr = args.lr * 0.1
    elif epoch >= 80:
        lr = args.lr * 0.01

    optimizer_P.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer_P.param_groups) - 1):
        optimizer_P.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch, wG):
    current_lr = adjust_learning_rate(optimizer_P, epoch)

    train_loss = AverageMeter()
    id_loss0 = AverageMeter()
    id_loss1 = AverageMeter()
    tri0_loss = AverageMeter()
    tri1_loss = AverageMeter()

    cg_loss = AverageMeter()
    ea_loss0 = AverageMeter()
    ea_loss1 = AverageMeter()

    correct = 0
    total = 0

    net.train()

    for batch_idx, (imgs_ir, imgs_ir_f, pids_ir, camid_ir, imgs_rgb, imgs_rgb_f, pids_rgb, camid_rgb) in enumerate(trainloader):

        input1 = imgs_rgb #(4,18,288,144)
        input2 = imgs_ir
        input3 = imgs_rgb_f#!
        input4 = imgs_ir_f

        label1 = pids_rgb
        label2 = pids_ir
        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        input3 = Variable(input3.cuda())#!
        input4 = Variable(input4.cuda())

        labels = Variable(labels.cuda())

        x_pool,out0,  f_pool,out1,  f_v2t_fir,f_t2v_fir,  f_v2t_sec,f_t2v_sec,  f_fir,f_sec = net(input1, input2, input3, input4, seq_len=seq_lenth)
        #池化，池化+降维， 池化，池化+降维， 重构v，重构i，原始f
        B = f_fir.size(0)

        loss_id0 = criterion1(out0, labels)
        loss_id1 = criterion1(out1, labels)

        loss_tri0, batch_acc0 = criterion2(x_pool, labels)
        loss_tri1, batch_acc1 = criterion2(f_pool, labels)

        loss_ea0 = criterion3(x_pool, labels)
        loss_ea1 = criterion3(f_pool, labels)

        loss_cg_fir = criterion_cg(f_v2t_fir,f_fir[B//2:].detach()) + criterion_cg(f_t2v_fir,f_fir[:B//2].detach())#添加！！！！
        loss_cg_sec = criterion_cg(f_v2t_sec,f_sec[B//2:].detach()) + criterion_cg(f_t2v_sec,f_sec[:B//2].detach())#不反向传播
        loss_cg     = loss_cg_fir + loss_cg_sec

        # Set the appropriate parameters according to your needs
        if epoch < 40 :
            loss_total = (loss_id0+loss_tri0+loss_ea0) + loss_tri1
        else :
            loss_total = (loss_id0+loss_tri0+loss_ea0) + (loss_id1+loss_tri1+loss_ea1) + loss_cg

        correct += (batch_acc0 / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        # optimization
        optimizer_P.zero_grad()
        loss_total.backward()
        optimizer_P.step()

        # log different loss components
        train_loss.update(loss_total.item(), 2 * input1.size(0))
        id_loss0.update(loss_id0.item(), 2 * input1.size(0))
        id_loss1.update(loss_id1.item(), 2 * input1.size(0))
        tri0_loss.update(loss_tri0.item(), 2 * input1.size(0))
        tri1_loss.update(loss_tri1.item(), 2 * input1.size(0))
        cg_loss.update(loss_cg.item(), 2 * input1.size(0))
        ea_loss0.update(loss_ea0.item(), 2 * input1.size(0))
        ea_loss1.update(loss_ea1.item(), 2 * input1.size(0))

        total += labels.size(0)


        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}] '                 
                  'i0: {id_loss0.val:.4f} ({id_loss0.avg:.4f}) '
                  'i1: {id_loss1.val:.4f} ({id_loss1.avg:.4f}) '
                  'T0: {tri0_loss.val:.4f} ({tri0_loss.avg:.4f}) '
                  'T1: {tri1_loss.val:.4f} ({tri1_loss.avg:.4f}) '
                  'cg: {cg_loss.val:.4f} ({cg_loss.avg:.4f}) '
                  'ea0: {ea_loss0.val:.4f} ({ea_loss0.avg:.4f}) '
                  'ea1: {ea_loss1.val:.4f} ({ea_loss1.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                   epoch, batch_idx, len(trainloader),

                   100. * correct / total,
                   train_loss = train_loss,
                   id_loss0 = id_loss0,
                   id_loss1 = id_loss1,
                   tri0_loss = tri0_loss,
                   tri1_loss = tri1_loss,
                   ea_loss0 = ea_loss0,
                   ea_loss1 = ea_loss1,
                   cg_loss = cg_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss0', id_loss0.avg, epoch)
    writer.add_scalar('id_loss1', id_loss1.avg, epoch)
    writer.add_scalar('tri0_loss', tri0_loss.avg, epoch)
    writer.add_scalar('tri1_loss', tri1_loss.avg, epoch)
    writer.add_scalar('cg_loss', cg_loss.avg, epoch)
    writer.add_scalar('ea_loss0', ea_loss0.avg, epoch)
    writer.add_scalar('ea_loss1', ea_loss1.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 1. / (1. + train_loss.avg)


def test2(epoch):
    net.eval()
    print('Extracting Gallery Feature...')
    ptr = 0
    gall_feat = np.zeros((ngall_1, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, imgs_f, pids, camids) in enumerate(galleryloader_1):
            input   = imgs
            input_f = imgs_f
            input   = Variable(input.cuda())
            input_f = Variable(input_f.cuda())

            batch_num = input.size(0)
            feat = net(input, input, input_f, input_f, test_mode[1], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    net.eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat = np.zeros((nquery_1, 2048))
    with torch.no_grad():
        for batch_idx, (imgs, imgs_f, pids, camids) in enumerate(queryloader_1):
            input   = imgs
            input_f = imgs_f
            input   = Variable(input.cuda())
            input_f = Variable(input_f.cuda())

            batch_num = input.size(0)
            feat = net(input, input, input_f, input_f, test_mode[0], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num
            q_pids.extend(pids)
            q_camids.extend(camids)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    distmat = np.matmul(query_feat, np.transpose(gall_feat))# compute the similarity
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)# evaluation
    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP


def test(epoch):
    net.eval()
    print('Extracting Gallery Feature...')
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, imgs_f, pids, camids) in enumerate(galleryloader):
            input   = imgs
            input_f = imgs_f
            input   = Variable(input.cuda())
            input_f = Variable(input_f.cuda())

            batch_num = input.size(0)
            feat    = net(input, input, input_f, input_f, test_mode[0], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    net.eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (imgs, imgs_f, pids, camids) in enumerate(queryloader):
            input   = imgs
            input_f = imgs_f
            input   = Variable(input.cuda())
            input_f = Variable(input_f.cuda())

            batch_num = input.size(0)
            feat    = net(input, input, input_f, input_f, test_mode[1], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num
            q_pids.extend(pids)
            q_camids.extend(camids)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    distmat = np.matmul(query_feat, np.transpose(gall_feat))# compute the similarity
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
    ranks = [1, 5, 10, 20]
    print("Results ----------")
    print("testmAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")
    return cmc,mAP


# training
print('==> Start Training...')

for epoch in range(start_epoch, 201):

    print('==> Preparing Data Loader...')
    sampler = IdentitySampler(data_set.ir_label, data_set.rgb_label, rgb_pos, ir_pos, args.num_pos, args.batch_size)
    index1  = sampler.index1
    index2  = sampler.index2

    loader_batch = args.batch_size * args.num_pos

    trainloader = DataLoader(
                  VideoDataset_train(data_set.train_ir, data_set.train_rgb, seq_len=seq_lenth, sample='video_train', transform=transform_train, index1=index1, index2=index2),
                  sampler=sampler, batch_size=loader_batch, num_workers=args.workers, drop_last=True,)

    # training
    wG = train(epoch, wG)

    if epoch >= 0 and epoch % 1 == 0:
        print('Test Epoch: {}'.format(epoch))
        print('Test Epoch: {}'.format(epoch), file=test_log_file)

        # testing
        cmc, mAP = test(epoch)

        if cmc[0] > best_acc:
            best_acc = cmc[0]
            best_t2v_rank1 = epoch
            state = {'net': net.state_dict(),
                     'mAP': mAP,
                     'epoch': epoch,}
            torch.save(state, checkpoint_path + suffix + 't2v_rank1_best.t')

        if mAP > best_map_acc:
            best_map_acc = mAP
            best_epoch = epoch
            state = {'net': net.state_dict(),
                     'mAP': mAP,
                     'epoch': epoch,}
            torch.save(state, checkpoint_path + suffix + 't2v_map_best.t')

        print('FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best t2v epoch [{}]'.format(best_t2v_rank1))
        print('FC(t2v):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
#-------------------------------------------------------------------------------------------------------------------
        cmc, mAP = test2(epoch)
        if cmc[0] > best_acc_v2t:
            best_acc_v2t = cmc[0]
            best_v2t_rank1 = epoch
            state = {'net': net.state_dict(),
                     'mAP': mAP,
                     'epoch': epoch,}
            torch.save(state, checkpoint_path + suffix + 'v2t_rank1_best.t')

        if mAP > best_map_acc_v2t:
            best_map_acc_v2t = mAP
            best_epoch = epoch
            state = {'net': net.state_dict(),
                     'mAP': mAP,
                     'epoch': epoch,}
            torch.save(state, checkpoint_path + suffix + 'v2t_map_best.t')

        print('FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print('Best v2t epoch [{}]'.format(best_v2t_rank1))
        print('FC(v2t):   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)

        test_log_file.flush()

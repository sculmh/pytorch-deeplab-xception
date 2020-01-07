import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

from dataloaders import make_test_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.metrics import Evaluator
from dataloaders.utils import decode_seg_map_sequence

class Test(object):

    def __init__(self, args):
        self.args = args

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.test_loader, self.nclass = make_test_data_loader(args, **kwargs)

        self.model = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)

        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        else:
            raise RuntimeError("=> pth file not specified")


    def test(self):
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        label_masks = []
        img_paths = []
        for i, sample in enumerate(tbar):
            image = sample['image']
            img_path = sample['img_path']
            img_paths.append(img_path)
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)  # [1, 513, 513]
            pred = pred.reshape(pred.shape[1], -1)
            label_masks.append(pred)
        # print('paths:', img_paths)
        rgb_masks = decode_seg_map_sequence(label_masks, dataset="cityscapes", is_test=True)
        for path, img in zip(img_paths, rgb_masks):
            path = path[0]
            if(path.find('leftImg8bit.png') == -1):
                raise RuntimeError('original name of each image should end with "leftImg8bit.png"')
            path = path.replace('leftImg8bit.png', 'seg.png')
            print('save path = ', path)
            # print('img.shape = ', img.shape)
            cv2.imwrite(path, np.array(img).transpose([1, 2, 0])) #[513, 513, 3]
        print('results already saved in leftImg8bit dir')


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='run/cityscapes/deeplab-xception/experiment_10/checkpoint.pth',
                        help='put the path to resuming file if needed')   # !important
    parser.add_argument('--checkname', type=str, default='deeplab-xception',
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    print('cuda is_available: ', torch.cuda.is_available())
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            print('using gpus: ', args.gpu_ids)
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    tester = Test(args)
    tester.test()

if __name__ == "__main__":
   main()





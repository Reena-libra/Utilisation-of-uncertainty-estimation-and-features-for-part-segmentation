"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
#from ShapeNetDataLoader1 import My_H5Dataset
from ShapeNetDataLoader2 import My_H5Dataset
from pathlib import Path
from tqdm import tqdm
#from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR2= '/home/user/reena'
#print(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

#seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
 #              'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
  #             'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_classes={'Wheat':[0,1]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def calculate_hybrid_scores(uncertainty, ear_counts, ear_ratios, alpha=1, beta=0, gamma=1):
    """
    Calculates hybrid scores combining uncertainty, ear count (feature importance), and ear ratio.
    All components are normalized and weighted before being combined.

    Parameters:
        uncertainty (Tensor): [B, N] - Per-point uncertainty
        ear_counts (Tensor): [B] - Ear count per sample
        ear_ratios (Tensor): [B] - Ear ratio per sample (between 0 and 1)
        alpha (float): Weight for uncertainty
        beta (float): Weight for ear count
        gamma (float): Weight for ear ratio

    Returns:
        hybrid_scores (Tensor): [B] - Final hybrid score per sample
    """

    # Default weights to avoid zero issues
    #default_uncertainty_weight = 0.0  # Uncertainty is already non-zero
    default_ear_count_weight = 0.5
    default_ear_ratio_weight = 0.01

    # Step 1: Normalize uncertainty
    max_uncertainty = uncertainty.max()
    #print('normalizeeu',uncertainty  / max_uncertainty)
    norm_uncertainty = uncertainty  / max_uncertainty if max_uncertainty > 0 else uncertainty
    #print('normalized_uncertsa',normalized_uncertainty)

    # 2. Replace zero ear counts with default
    safe_ear_counts = torch.where(
        ear_counts == 0,
        torch.tensor(default_ear_count_weight, device=ear_counts.device, dtype=ear_counts.dtype),
        ear_counts
    )
    max_ear_count = safe_ear_counts.max()
    norm_ear_counts = safe_ear_counts / max_ear_count if max_ear_count > 0 else safe_ear_counts

    # 3. Replace zero ear ratios with default
    safe_ear_ratios = torch.where(
        ear_ratios == 0,
        torch.tensor(default_ear_ratio_weight, device=ear_ratios.device, dtype=ear_ratios.dtype),
        ear_ratios
    )
    max_ear_ratio = safe_ear_ratios.max()
    norm_ear_ratios = safe_ear_ratios / max_ear_ratio if max_ear_ratio > 0 else safe_ear_ratios

    # 4. Combine all with weights
    hybrid_scores = alpha * norm_uncertainty + beta * norm_ear_counts + gamma * norm_ear_ratios

    return hybrid_scores


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch Size during training')
    parser.add_argument('--epoch', default=150, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg_g2re_m3_extended')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    TRAIN_DATASET=My_H5Dataset(os.path.join(BASE_DIR2, 'wheat_g2re2k/TRAIN_G2RE2K_Wheat.h5'),normal_channel=False)
    TEST_DATASET=My_H5Dataset(os.path.join(BASE_DIR2, 'wheat_g2re2k/VAL_G2RE2K_Wheat.h5'),normal_channel=False)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    #TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))


    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/GAM_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    #criterion = MODEL.get_loss().cuda()
    criterion2 = MODEL.get_loss_new().cuda()
    criterion3=MODEL.CalculateUncertaintyLogits2().cuda()
    criterion4=MODEL.calculate_density().cuda()
    
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    best_cat0_avg_iou=0      #####
    best_cat1_avg_iou=0      #######


    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target, ears) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target, ears = points.float().cuda(), label.long().cuda(), target.long().cuda(), ears.long().cuda()
            points = points.transpose(2, 1)
            
           
            
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            uncertainty = criterion3.mc_dropout_variance(classifier, points, to_categorical(label, num_classes), num_samples=5)
            
            
            tar=target
            #print(tar.shape)
            feature_ratio=criterion4(tar) #####################
            
            #print(uncertainty,ears)
            feature_importance_scores = ears
            
            hybrid_scores = calculate_hybrid_scores(uncertainty, feature_importance_scores,feature_ratio)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            
            
            
            #loss = criterion(seg_pred, target, trans_feat)
            loss = criterion2(seg_pred, target.long(),hybrid_scores,batch_size=args.batch_size,num_points=points.shape[2])
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            shape_ious1 = {cat: [] for cat in seg_classes.keys()}
            shape_ious2 = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target, ears) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target, ears = points.float().cuda(), label.long().cuda(), target.long().cuda(), ears.long().cuda()
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))
                    shape_ious1[cat].append(np.mean(part_ious[0]))  ##new                                     
                    shape_ious2[cat].append(np.mean(part_ious[1]))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
                shape_ious1[cat]=np.mean(shape_ious1[cat])    ###############
                shape_ious2[cat]=np.mean(shape_ious2[cat])  #################

            mean_shape_ious = np.mean(list(shape_ious.values()))
            mean_shape_ious1=np.mean(list(shape_ious1.values())) ###########           
            mean_shape_ious2=np.mean(list(shape_ious2.values())) ################
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            ################################################## part id mean           ###############
            test_metrics['class_cat0_avg_iou']=mean_shape_ious1                ##############
            test_metrics['class_cat1_avg_iou']=mean_shape_ious2  
            
  ##########################################################################################
            

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f  Cat0 avg mIOU: %f    Cat1 avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou'],test_metrics['class_cat0_avg_iou'],test_metrics['class_cat1_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                 
                'class_cat0_avg_iou': test_metrics['class_cat0_avg_iou'],        
                'class_cat1_avg_iou': test_metrics['class_cat1_avg_iou'],      

                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        if test_metrics['class_cat0_avg_iou'] > best_cat0_avg_iou:               #######
             best_cat0_avg_iou = test_metrics['class_cat0_avg_iou']
        if test_metrics['class_cat1_avg_iou'] > best_cat1_avg_iou:
             best_cat1_avg_iou = test_metrics['class_cat1_avg_iou']


        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        
        log_string('Category 0 is: %.5f' % best_cat0_avg_iou)
        log_string('Category 1 is: %.5f' % best_cat1_avg_iou)
        
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        logger.info('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)

        logger.info('Category 0 is: %.5f' % best_cat0_avg_iou)
        logger.info('Category 1 is: %.5f' % best_cat1_avg_iou)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)

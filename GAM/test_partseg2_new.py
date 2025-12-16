"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os

import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
#from mayavi import mlab
#import open3d as o3d
#from data_utils.ShapeNetDataLoader2 import My_H5Dataset
from ShapeNetDataLoader1 import My_H5Dataset
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
BASE_DIR2= '/home/user/reena'

results_dir=os.path.join(ROOT_DIR,'CM')
#seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
 #              'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
  #             'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
   #            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_classes = {'Wheat' :[0,1]}
print('seg_classes',seg_classes)  
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg_p2re_m6_extended/' + args.log_dir


    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    #TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    TEST_DATASET = My_H5Dataset(os.path.join(BASE_DIR2, 'wheat_p2re2k/TEST_P2RE2K_Wheat.h5'),normal_channel=False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print('model_name',model_name,experiment_dir)
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
   
    #print('classifier',classifier.convs1)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()


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
        
        all_preds = [] ####precision
        all_gts = []    ###recall

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        #m=1
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()


                ####for saliency

           # print('target',target.shape)
            
            
            points1=points.clone().detach().requires_grad_(True)
            points = points.transpose(2, 1)
            #points1.requires_grad_()
           
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            
            for _ in range(args.num_votes):
                seg_pred,_ = classifier(points, to_categorical(label, num_classes))
               # print('seg_pred',seg_pred.shape)
                #seg_pred= classifier(points, to_categorical(label, num_classes))
                #print('seg_pred',seg_pred.shape)
                #print('f',f.shape)
                vote_pool += seg_pred
########################################################################################
            #seg_pred1 = seg_pred.contiguous().view(-1, num_part)
            #target1 = target.view(-1, 1)[:, 0]
            #pred_choice = seg_pred.data.max(1)[1]
            #print('loss seg and target',seg_pred1.shape,target1.shape,label.shape)
            #seg_pred1.requires_grad=True


            #target.requires_grad=True
            #print("seg_pred.requires_grad:", seg_pred1.requires_grad)
            #print("target.requires_grad:", target1.requires_grad)
            #criterion = MODEL.get_loss().cuda()
            #loss = criterion(seg_pred1, target1,_)
            #print('loss',loss)
            #with torch.enable_grad():
            #loss.requires_grad = True
            #loss.backward()
            #seg_pred1.backward()
            #optimizer.step()


            # Compute saliency map

            #print('fjfj',points1.grad,points1.shape)
            
            #gradients = points1.grad
            #print('gradients',gradients)
                  # Use the saliency map for further analysis or visualization
            
            #seg_pred1.backward(gradient=torch.ones_like(seg_pred1)) 
            #saliency_map = torch.abs(points1.grad).max(dim=1)[0].squeeze()
            #print('saliency map',saliency_map) 
            #saliency_map = torch.abs(points1.grad).max(dim=1)[0].squeeze() 
              
            #seg_pred1.requires_grad = False
            #points1.grad.data.zero_()


            #########################################################################
          
            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
           # print('cur_pred_val_logits',cur_pred_val_logits.shape)
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
               # print('cat',cat)
                logits = cur_pred_val_logits[i, :, :]
                #print('logits',logits.shape)
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

           # print('cur_pred_val',cur_pred_val.shape)
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                
                all_preds.append(cur_pred_val.flatten())   ####precision
                all_gts.append(target.flatten())              ###recall
               # print('segp and segl',segp.shape,segl.shape)
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
                   
                    
                shape_ious2[cat].append(np.mean(part_ious[1]))  ##new
                    #print('2 cat',shape_ious2[cat])
            
                ####### new code for salinecy map#####
               
            
            
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        mean_shape_ious1=np.mean(list(shape_ious1.values())) ###########
           
        mean_shape_ious2=np.mean(list(shape_ious2.values())) ################

        test_metrics['accuracy'] = total_correct / float(total_seen)
        #test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        ################################################## part id mean           ###############
        test_metrics['class_cat0_avg_iou']=mean_shape_ious1                ##############
        test_metrics['class_cat1_avg_iou']=mean_shape_ious2   
       
        
        
        all_preds_flat = np.concatenate(all_preds)  # shape: (B * N,)
        all_gts_flat = np.concatenate(all_gts)
        
        
        precision = precision_score(all_gts_flat, all_preds_flat, average='macro', zero_division=0)
        recall = recall_score(all_gts_flat, all_preds_flat, average='macro', zero_division=0)
        f1 = f1_score(all_gts_flat, all_preds_flat, average='macro', zero_division=0)
        cm = confusion_matrix(all_gts_flat, all_preds_flat)
        
        test_metrics['precision'] = precision 
        test_metrics['recall'] = recall
        test_metrics['f1_score'] = f1
        
        
  ##########################################################################################

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])

    log_string('Cat 0 avg mIOU is: %.5f' % test_metrics['class_cat0_avg_iou'])
    log_string('Cat 1 avg mIOU is: %.5f' % test_metrics['class_cat1_avg_iou'])
    
    log_string('Precision: %.5f' % precision)
    log_string('Recall: %.5f' % recall)
    log_string('F1-score: %.5f' % f1)
    log_string('Confusion Matrix:\n%s' % str(cm))
    cm_normalized = confusion_matrix(all_gts_flat, all_preds_flat, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Normalized Confusion Matrix')
    #plt.savefig((os.path.join(results_dir, 'confusion_matrix_normalized_a_w.png')))
    plt.close()






if __name__ == '__main__':
    args = parse_args()
    main(args)

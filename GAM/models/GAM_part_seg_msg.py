import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GAM_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        #self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=135+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
    
class CalculateUncertaintyLogits2(nn.Module):
    def __init__(self):
        super(CalculateUncertaintyLogits2, self).__init__()

    def mc_dropout_variance(self, model, inputs, cls_label, num_samples=5):
        model.train()  # Ensure dropout is active during inference
        predictions = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Perform forward pass with inputs and class labels
                seg_pred, _ = model(inputs, cls_label)  # seg_pred: [batch_size, num_points, num_classes]
                predictions.append(seg_pred.unsqueeze(0))  # Add a new dimension for stacking
        
        # Stack predictions across num_samples
        predictions = torch.cat(predictions, dim=0)  # Shape: [num_samples, batch_size, num_points, num_classes]
        #print('pred',predictions.shape)
        # Calculate mean and variance across MC-Dropout samples (across num_samples dimension)
        variance_pred = predictions.var(dim=0)  # Variance of predictions: [batch_size, num_points, num_classes]
        mean_variance_per_point = variance_pred.mean(dim=[2])  # Shape: [batch_size]
        #print(mean_variance_per_point)
        
        #print(inputs.shape[0])
         # Visualize per-point uncertainty for each sample in the batch
        #for i in range(inputs.shape[0]):  # Loop over each sample in the batch
            #print(inputs.shape)
            
            #filename = f"uncertainty_batch_sample_{i}.png"
            #print(inputs.shape,variance_pred.shape)
            
            #self.visualize_uncertainty(inputs[i, :,:], mean_variance_per_point[i,:],i)  #                    Visualize per-point uncertainty
            
            
        # Calculate the mean variance for each sample by averaging over points and classes
        mean_variance_per_sample = variance_pred.mean(dim=[1, 2])  # Shape: [batch_size]
        
        return mean_variance_per_sample
    
    def visualize_uncertainty(self, coordinates, uncertainty,i):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,8))
        #fig = plt.figure(figsize=(12, 6))
        #coordinates = coordinates.cpu().numpy()  # Move points tensor to CPU and convert to NumPy
        print(coordinates.shape)
        #uncertainty = uncertainty.cpu().numpy()
        ground_truth = ['green', 'red']
       
    # Create a 3D subplot for the uncertainty visualization
        #ax1 = fig.add_subplot(121, projection='3d')
        '''
        for j in range( 2048):
        
            x=coordinates[0,j].cpu().data.numpy()
            y=coordinates[1,j].cpu().data.numpy()
            z=coordinates[2,j].cpu().data.numpy()
            u=int(uncertainty[j].cpu().data.numpy())
            #print(x,y,z)
            
            ax[0].scatter(x,y,z, c=u, cmap='viridis')
            ax[0].set_title('Uncertainty Visualization')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Y')
            #ax[0].set_zlabel('Z')
        '''
        #plt.savefig(filename)
       # plt.savefig(f'result_plot_{i}.png')
        plt.clf()
        print('yes')
        #plt.close()

class get_loss_new(nn.Module):
    def __init__(self):
        super(get_loss_new, self).__init__()

    def forward(self, pred, gold, weights=None,batch_size=None,num_points=None, smoothing=True):
        #print('pred',pred.shape,gold.shape)
        gold = gold.contiguous().view(-1)  # Flatten target labels for calculation
       
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            # One-hot encoding for label smoothing
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            # Calculate per-sample loss with label smoothing
            per_sample_loss = -(one_hot * log_prb).sum(dim=1)
            #print(per_sample_loss.shape)
            #per_sample_loss = -(one_hot * log_prb).sum(dim=2)
        else:
            # Standard cross-entropy without label smoothing
            per_sample_loss = F.cross_entropy(pred, gold, reduction='none')  # per-sample loss
            #per_sample_loss = per_sample_loss.view(batch_size, num_points)  # Reshape to [batch_size, num_points]
        
        per_sample_loss = per_sample_loss.view(batch_size, num_points)  # Shape: [batch_size, num_points]
        
        # Average over points within each sample to get per-sample loss
        per_sample_loss = per_sample_loss.mean(dim=1)  # Shape: [batch_size]
        #per_sample_loss = per_point_loss.mean(dim=1)
        # Apply weights if provided
        if weights is not None:
            #print('persample',per_sample_loss.shape)
            #per_sample_loss=per_sample_loss.view(num_points,batch_size)
            #print('loss',per_sample_loss.shape, weights.shape)
            per_sample_loss = per_sample_loss * weights  # Element-wise multiplication with weights

        # Return mean of weighted losses
        return per_sample_loss.mean()


    


    
class calculate_density(nn.Module):
    def __init__(self):
        super(calculate_density, self).__init__()

    def forward(self, labels_batch):
        """
        Compute density-based weights based on the number of 'ear' points (label == 1) in each sample.

        Args:
            labels_batch: Tensor of shape [B, N] where B is batch size and N is number of points

        Returns:
            class_weights: Tensor of shape [B] with weights per sample
        """
        batch_size = labels_batch.size(0)
        class_weights = []

        for i in range(batch_size):
            labels = labels_batch[i, :]  # Shape: [N]
            num_ear_points = torch.sum(labels == 1)
            total_points = labels.numel()

            density_ratio = torch.true_divide(num_ear_points, total_points)  # scalar
            cls_weight = 1.0 / (1.0 + torch.exp(-density_ratio))  # Apply sigmoid
            class_weights.append(cls_weight)

        return torch.stack(class_weights)  # Return as Tensor of shape [B]

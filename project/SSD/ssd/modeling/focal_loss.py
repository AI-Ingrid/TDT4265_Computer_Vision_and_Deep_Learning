import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
    

class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.alpha = 0.01
        self.gamma = 0.01

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def one_hot_encode(self, to_encode):
        encoded = F.one_hot(to_encode)
        return encoded

    def compute_focal_loss(self, output, labels):
        """
        Args: 
            output: Shape [batch_size, num_classes_ num_boxes] 
            labels: Targets image of shape [batch_size, num_boxes]
        Return:
            focal_loss: The focal loss (float)
        """
        #print('Ouput ',output)
        #print('Ouput shape ',output.shape)
        # Get softmax probability for each classes
        output_probs = F.softmax(output, dim=1).cpu().detach().numpy()
        output_probs = np.moveaxis(output_probs, -1, 1)

        
        # One-hot encode the labels
        labels = F.one_hot(labels).cpu().detach().numpy()

        # TODO LOG 10 ?
        # k = 0
        # (32, 65440, 9)
        print("Labes shape", labels.shape)
        # (32, 9, 65440) -> T : (9, 65440, 32)
        print("Prob shape", np.log10(output_probs).shape)
        # 0.01 * 1 - (32, 9, 65440)^0.01   
        matrise_1 = self.alpha * (1-output_probs)**self.gamma
        # (32, 65440, 9) @ (32, 9, 65440) = (32, 65440, 65440)
        matrise_2 = labels * np.log10(output_probs)
        print("matrise1 shape:", matrise_1.shape)
        print("matrise2 shape:", matrise_2.shape)
        loss = - np.sum(matrise_1 * matrise_2)
        # k > 1
        self.alpha = 1
        self.gamma = 1
        # Cn = -torch.sum(alpha*(1-soft)**gamma*targets*log_soft)
        loss += -self.alpha * ((1-output_probs)**self.gamma * labels) * np.log10(output_probs)
        print("loss type new: ", type(torch.from_numpy(loss).cuda()))
        loss = torch.sum(torch.from_numpy(loss).cuda())
        return loss.mean()

        """
        KOK
        """
        targets = F.one_hot(targets, 9) # Transform targets from [batch size, num_boxes] to [batch size, num_boxes, num_categories]
    
        log_soft = torch.permute(F.log_softmax(outputs, dim=1),(0, 2, 1))
        soft = torch.permute(F.softmax(outputs, dim=1),(0, 2, 1))
        assert targets.shape == log_soft.shape,\
            f"Targets shape: {targets.shape}, outputs: {log_soft.shape}"

        Cn = -torch.sum(alpha*(1-soft)**gamma*targets*log_soft)
        C = Cn/(targets.size(dim=1))
        return C


    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """

        """
        Cross-entropy 
        
        with torch.no_grad():
            to_log = - F.log_softmax(confs, dim=1)[:, 0]
            mask = hard_negative_mining(to_log, gt_labels, 3.0)
            print("size mask: ", mask.size())
        
        
        classification_loss2 = F.cross_entropy(confs, gt_labels, reduction="none")
        #print("Classification loss 1 shape:", classification_loss.size())
        classification_loss2 = classification_loss2[mask].sum()
        
        """
        

        classification_loss = self.compute_focal_loss(confs, gt_labels)
        total_loss = 0
        to_log = 0 

        
        """
        La det under stå
        """

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + classification_loss/num_pos
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss/num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log

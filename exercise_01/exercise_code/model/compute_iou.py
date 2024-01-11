import torch


def compute_iou(bbox_1: torch.Tensor, bbox_2: torch.Tensor) -> torch.Tensor:
    assert bbox_1.shape == bbox_2.shape

    ########################################################################
    # TODO:                                                                #
    # Compute the intersection over union (IoU) for two batches of         #
    # bounding boxes, each of shape (B, 4). The result should be a tensor  #
    # of shape (B,).                                                       #
    # NOTE: the format of the bounding boxes is (ltrb), meaning            #
    # (left edge, top edge, right edge, bottom edge). Remember the         #
    # orientation of the image coordinates.                                #
    # NOTE: First calculate the intersection and use this to compute the   #
    # union                                                                #
    # iou = ...                                                            #
    ########################################################################
    def getArea(left,top,right,bottom): return torch.clip(right - left,min=0) * torch.clip(bottom-top,min=0)
        
    left_inter   = torch.max(bbox_1[:,0],bbox_2[:,0])
    top_inter    = torch.max(bbox_1[:,1],bbox_2[:,1])
    right_inter  = torch.min(bbox_1[:,2],bbox_2[:,2])
    bottom_inter = torch.min(bbox_1[:,3],bbox_2[:,3])
    
    bbox_1_area  = getArea(bbox_1[:,0],bbox_1[:,1],bbox_1[:,2],bbox_1[:,3])
    bbox_2_area  = getArea(bbox_2[:,0],bbox_2[:,1],bbox_2[:,2],bbox_2[:,3])
    intersection = getArea(left_inter,top_inter,right_inter,bottom_inter)
    union = bbox_1_area + bbox_2_area - intersection
    iou = (intersection)/(union)
    
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return iou

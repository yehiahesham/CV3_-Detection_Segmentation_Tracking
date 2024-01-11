import torch

from exercise_code.model.compute_iou import compute_iou

def non_maximum_suppression(bboxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    ########################################################################
    # TODO:                                                                #
    # Compute the non maximum suppression                                  #
    # Input:                                                               #
    # bounding boxes of shape B,4                                          #
    # scores of shape B                                                    #
    # threshold for iou: if the overlap is bigger, only keep one of the    #
    # bboxes                                                               #
    # Output:                                                              #
    # bounding boxes of shape B_,4                                         #
    ########################################################################
    #APPROACH 1
    # bboxes_nms=[]
    # for i in range(bboxes.shape[0]):
    #     discard=False
    #     for j in range(i+1,bboxes.shape[0]): 
    #         iou = compute_iou(bboxes[i].unsqueeze(0),bboxes[j].unsqueeze(0)).item()
    #         if iou > threshold and bool(scores[i]<scores[j].item()): 
    #             discard=True
    #             break
    #     if not discard: bboxes_nms.append(bboxes[i])
    # bboxes_nms = torch.stack(bboxes_nms)
    
    #APPROACH 2 
    # I don;t know why is this failing the test !?
    toKeep=[]
    _,scores_ordered_indices = scores.sort(descending=True)
    while scores_ordered_indices.numel() > 0:
        idx = scores_ordered_indices[0]
        toKeep.append(idx)
        
        if scores_ordered_indices.numel() == 1:  break #no other boxes to compare to , Finised!
        
        other_boxes = bboxes[scores_ordered_indices[1:]]
        ious = torch.Tensor([compute_iou(bboxes[idx].unsqueeze(0),other_box.unsqueeze(0)).item() for other_box in other_boxes])
        scores_ordered_indices = scores_ordered_indices[1:][ious <= threshold] #filter out the rejected
    
    bboxes_nms = bboxes[torch.tensor(toKeep)]


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return bboxes_nms

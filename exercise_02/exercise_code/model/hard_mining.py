import torch
from exercise_code.model.distance_metrics import euclidean_squared_distance


class HardBatchMiningTripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining of samples in a batch.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        # super(HardBatchMiningTripletLoss, self).__init__()
        super().__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def compute_distance_pairs(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        Output:
            distance_positive_pairs (torch.Tensor): distance (to one pos. sample per sample) with shape (batch_size).
            distance_negative_pairs (torch.Tensor): distance (to one neg. sample per sample) with shape (batch_size).
        """
        batch_size = inputs.size(0)

        ########################################################################
        # TODO:                                                                #
        # Compute the pairwise euclidean distance between all n feature        #
        # vectors. You can reuse your implementation, which we already         #
        # imported to this file.                                               #
        # NOTE: We recommend computing the actual euclidean distance (not      #
        # squared).For numerical stability, you can do something like:         #
        # distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()            #
        #                                                                      #
        # distance_matrix = ...                                                #
        ########################################################################
       
        # distmat = torch.zeros((batch_size, batch_size), dtype=inputs.dtype, device=inputs.device)
        # chunk_size=20
        # for i in range(0, batch_size, chunk_size):
        #     inputs_chunk = inputs[i:i + chunk_size]
            
        #     for j in range(0, batch_size, chunk_size):
        #         input2_chunk = inputs[j:j + chunk_size]
                    
        #         distances = inputs_chunk.unsqueeze(1) - input2_chunk.unsqueeze(0)
        #         distances = distances.clamp(min=1e-12)
        #         distances = distances **2
        #         distmat[i:i + chunk_size, j:j + chunk_size] = torch.sum(distances,dim=2)
        
        distmat = euclidean_squared_distance(inputs, inputs)
        distmat = distmat.clamp(min=1e-12).sqrt()
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        ########################################################################
        # TODO:                                                                #
        # For each sample, find the hardest to match positive and hardest      #
        # to distinguish negative sample.                                      #
        # The ground truth classes are stored in "targets", which stores a     #
        # label for each of the samples. Pairs of samples with the SAME        #
        # class form a positive sample. Pairs of samples with a DIFFERENT      #
        # class form a negative sample.                                        #
        #                                                                      #
        # For this task, you will need to loop over all samples in the batch,  #
        # and for each one find the hardest positive/negative samples          #
        # in the batch.                                                        #
        # The distances are then added to the following lists.                 #
        # Think about what hardest means for positive and negative pairs.      #
        # NOTE: Positive pairs should be as close as possible, while negative  #
        # pairs should be quite far apart.                                     #
        #                                                                      #
        # distance_positive_pairs, distance_negative_pairs = [], []            #
        # for i in range(batch_size):                                          #
        #     distance_positive_pairs.append(...)                              #
        #     distance_negative_pairs.append(...)                              #
        ########################################################################

        # hardest negative: an image that looks VERY    close to the Anchor and should be very very hard to NOT match to the same class
        # Hardest Postive : an image that looks NOTHING close to the Anchor and should be very very hard to     match to the same class.

        # Pairs of samples with the SAME class form a positive sample.
            # Positive pairs should be as close as possible

        # Pairs of samples with the DIFF class form a negative sample.
            # negative pairs should be quite far apart.


        # loop over all samples in the batch
            # for each one find the hardest positive/negative samples in the batch.
        # distance_matrix_max = torch.max(distmat,dim=1)
        # distance_matrix_min = torch.max(distmat,dim=1)
        
        
        # It compares each element in the first dimension of the first tensor with each element in the second dimension of the second tensor.
        # creates a boolean matrix where each element (i, j) is True if labels[i] == labels[j] and False otherwise.
        # same_class_mask = targets.unsqueeze(0) == targets.unsqueeze(1)
        # diff_class_mask = ~same_class_mask
        
        # # distance_positive_pairs, distance_negative_pairs = [], [] 
        # distance_positive_pairs,distance_negative_pairs = torch.zeros(batch_size),torch.zeros(batch_size)


        # for i in range(len(same_class_mask)):
        #     same_distt,diff_distt=[],[]
        #     for j in range(len(same_class_mask)): 
        #         if same_class_mask[i][j]:  same_distt.append(distmat[i][j])
        #         else: diff_distt.append(distmat[i][j])
            
        #     hardest_postive = max(same_distt)
        #     hardest_negative= min(diff_distt)
        #     distance_positive_pairs.append(hardest_postive)
        #     distance_negative_pairs.append(hardest_negative)
        #     print(f'hardest -ve (min dis) = {hardest_negative}, while hardest +ve (max dis) = {hardest_postive}')
        
        same_class_mask = targets.unsqueeze(0) == targets.unsqueeze(1)
        diff_class_mask = ~same_class_mask
        distance_positive_pairs,distance_negative_pairs = torch.zeros(batch_size),torch.zeros(batch_size)
        for i in range(batch_size):            
            # Extract distances for the current index
            hardest_positive = distmat[i][same_class_mask[i]]
            hardest_negative = distmat[i][diff_class_mask[i]]
            
            distance_positive_pairs[i] = (hardest_positive.max())
            distance_negative_pairs[i] = (hardest_negative.min())
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        ########################################################################
        # TODO:                                                                #
        # Convert the created lists into 1D pytorch tensors. Please never      #
        # convert the tensors to numpy or raw python format, as you want to    #
        # backpropagate the loss, i.e., the above lists should only contain    #
        # pytorch tensors.                                                     #
        # NOTE: Checkout the pytorch documentation.                            #
        #                                                                      #
        # distance_positive_pairs = ...                                        #
        # distance_negative_pairs = ...                                        #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return distance_positive_pairs, distance_negative_pairs

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        Output:
            loss (torch.Tensor): scalar loss, reduction by mean along the batch_size.
        """

        distance_positive_pairs, distance_negative_pairs = self.compute_distance_pairs(inputs, targets)
        distance_positive_pairs = distance_positive_pairs.to(inputs.device)
        distance_negative_pairs = distance_negative_pairs.to(inputs.device)
        # The ranking loss will compute the triplet loss with the margin.
        # loss = max(0, -1*(neg_dist - pos_dist) + margin)
        # This is done already, no need to change anything.
        y = torch.ones_like(distance_negative_pairs)
        # one in y indicates, that the first input should be ranked higher than the second input, which is true for all the samples
        return self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)

class CombinedLoss(object):
  def __init__(self, margin=0.3, weight_triplet=1.0, weight_ce=1.0):
      super(CombinedLoss, self).__init__()
      self.triplet_loss = HardBatchMiningTripletLoss() # <--- Your code is used here!
      self.cross_entropy = torch.nn.CrossEntropyLoss()
      self.weight_triplet = weight_triplet
      self.weight_ce = weight_ce

  def __call__(self, logits, features, gt_pids):
      loss = 0.0
      loss_summary = {}
      if self.weight_triplet > 0.0:
        loss_t = self.triplet_loss(features, gt_pids) * self.weight_triplet
        loss += loss_t
        loss_summary['Triplet Loss'] = loss_t
      
      if self.weight_ce > 0.0:
        loss_ce = self.cross_entropy(logits, gt_pids) * self.weight_ce
        loss += loss_ce
        loss_summary['CE Loss'] = loss_ce

      loss_summary['Loss'] = loss
      return loss, loss_summary
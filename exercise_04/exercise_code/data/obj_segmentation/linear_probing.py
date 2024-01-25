import torch
import torch.nn as nn


class LinearProbingNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        ########################################################################
        # TODO:                                                                #
        # Define a ONE layer neural network with a 1x1 convolution and a       #
        # sigmoid non-linearity to do binary classification on an image        #
        # NOTE: the network receives a batch of feature maps of shape          #
        # B x H x W x input_dim and should output a binary classification of   #
        # shape B x H x W                                                      #
        ########################################################################


        self.classification_head = nn.Sequential(
            nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
            )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x is a batch of feature maps of shape B x H x W x feat_dim

        ########################################################################
        # TODO:                                                                #
        # Do the forward pass of you defined network                           #
        # prediction = ...                                                     #
        ########################################################################


        return self.classification_head(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return prediction

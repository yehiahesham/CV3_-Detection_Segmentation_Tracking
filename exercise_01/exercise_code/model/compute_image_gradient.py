from typing import Tuple
import torch.nn as nn
import torch

APPROACH=1
def compute_image_gradient(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # images B x H x W

    ########################################################################
    # TODO:                                                                #
    # Compute the 2-dimenational gradient for a given grey image of size   #
    # B x H x W. The return values of this function should be the norm and #
    # the angle of this gradient vector.                                   #
    # NOTE: first, calculate the gradient in x and y direction             #
    # (you will need add padding to the image boundaries),                 #
    # then, compute the vector norm and angle.                             #
    # The angle of a given gradient angle is defined                       #
    # in degrees (range=0.,,,.360).                                        #
    # NOTE: The angle is defined counter-clockwise angle between the       #
    # gradient and the unit vector along the x-axis received from atan2.   #
    ########################################################################
        images = images.unsqueeze(1) 
        # Scharr_x=torch.tensor([[3,0,-3],[10,0,-10],[3,0,-3]],dtype=torch.float32)
        Scharr_x=torch.tensor([[-47, 0, 47], [-162, 0, 162], [-47, 0, 47]],dtype=torch.float32) / 320.0
        Scharr_y= Scharr_x.T.view(1, 1, 3, 3)
        Scharr_x = Scharr_x.view(1, 1, 3, 3)
        
        gx = nn.functional.conv2d(images,Scharr_x,padding="same")
        gy = nn.functional.conv2d(images,Scharr_y,padding="same")
        gx=gx.squeeze(1)
        gy=gy.squeeze(1)
        norm  = torch.sqrt(gx ** 2 +gy  ** 2)
        angle = torch.atan2(gx, gy)
        angle = torch.rad2deg(angle)
        angle = torch.where(angle < 0, angle+360, angle)

        # padded_images = nn.functional.pad(images, (1, 1, 1, 1), mode="replicate")
        # x_grads = torch.zeros_like(images)
        # y_grads = torch.zeros_like(images)

        # x_grads[:, :, :] = padded_images[:, 1:-1, 2:] - padded_images[:, 1:-1, :-2]
        # y_grads[:, :, :] = padded_images[:, :-2, 1:-1] - padded_images[:, 2:, 1:-1]

        # norm = torch.sqrt(x_grads**2 + y_grads**2)
        # angle = torch.atan2(x_grads, y_grads)
        # angle = torch.rad2deg(angle)
        # angle = torch.where(angle < 0, angle+360, angle)



    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    
        return norm, angle

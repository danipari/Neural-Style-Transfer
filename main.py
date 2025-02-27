import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import server
server.start_server_in_background()

class Animation():
    def __init__(self):
        self.frames = []

    def append(self, frame):
        self.frames.append(frame)

    def save(self, path):
        self.frames[0].save(path, format='GIF', append_images=self.frames[1:], save_all=True, duration=100, loop=0)


class VggNet(torch.nn.Module):
    def __init__(self, num_classes=1000, vgg=16):
        """
        VGGNet implementation for image classification.

        Args:
            num_classes (int, optional): Number of output classes. Default is 1000 (for ImageNet).
            vgg (int, optional): VGG configuration, either 11, 13, 16 or 19 for VGG-11, VGG-16 or VGG-19. 
                                    Default is 19.
        """
        super(VggNet, self).__init__()
        self.num_classes = num_classes
        self.activation  = dict()

        if vgg not in (11, 13, 16, 19):
            raise ValueError("vgg must be 11, 13, 16, or 19")

        # Define the number of convolutional layers per block based on the VGG variant.
        # Canonical configurations:
        # VGG-11: [1, 1, 2, 2, 2]
        # VGG-13: [2, 2, 2, 2, 2]
        # VGG-16: [2, 2, 3, 3, 3]
        # VGG-19: [2, 2, 4, 4, 4]
        if vgg == 11:
            conv_counts = [1, 1, 2, 2, 2]
        elif vgg == 13:
            conv_counts = [2, 2, 2, 2, 2]
        elif vgg == 16:
            conv_counts = [2, 2, 3, 3, 3]
        else:  # vgg == 19
            conv_counts = [2, 2, 4, 4, 4]

        # Build convolutional blocks 
        self.block1 = self._create_conv_block(in_channels=3,   out_channels=64,  num_convs=conv_counts[0])
        self.block2 = self._create_conv_block(in_channels=64,  out_channels=128, num_convs=conv_counts[1])
        self.block3 = self._create_conv_block(in_channels=128, out_channels=256, num_convs=conv_counts[2])
        self.block4 = self._create_conv_block(in_channels=256, out_channels=512, num_convs=conv_counts[3])
        self.block5 = self._create_conv_block(in_channels=512, out_channels=512, num_convs=conv_counts[4])


    def _create_conv_block(self, in_channels, out_channels, num_convs):
        """
        Create a convolutional block as:
        [num_convs x (Conv2d -> ReLU)] -> MaxPool2d

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_convs (int): Number of convolutional layers in the block.
        Returns:
            nn.Sequential: The convolutional block.
        """
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # the next convolution uses out_channels as input

        layers.append(nn.AvgPool2d(kernel_size=2, stride=2)) # Modification wrt original version
        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
    
    # From https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook
    # a dict to store the activations
    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output
        return hook


    def feature_maps(self, x, content_layers=[], style_layers=[], detach=False):
        content_feature_map = {}
        style_feature_map   = {}
        self.activation     = {}
        hooks               = []

        # Attach hook to all layer
        layers = content_layers + style_layers
        for layer in layers:
            n_block, n_layer = layer[:6], 2*(int(layer[7])-1)
            hook = self.get_submodule(n_block)[n_layer].register_forward_hook(self.getActivation(layer))
            hooks.append(hook) 

        # Perform forward pass
        self.forward(x)

        # Extact feature maps
        for layer in content_layers:
            content_feature_map[layer] = self.activation[layer].detach() if detach else self.activation[layer]

        for layer in style_layers:
            style_feature_map[layer] = self.activation[layer].detach() if detach else self.activation[layer]

        for hook in hooks:
            hook.remove()

        if content_layers and not style_layers:
            return content_feature_map
        elif style_layers and not content_layers:
            return style_feature_map
        else:
            return content_feature_map, style_feature_map
    
    
# transpose is use to meet the shape of the tensor C x H x W rather than H x W x C
mean_pixels = torch.from_numpy(np.float32(np.load('data/mean_pixels.npy').transpose(2, 0, 1)))

def meanSubstraction(x):
    return x - transforms.Resize(x.shape[1:])(mean_pixels)

def meanAddition(x):
    return torch.clip(x + transforms.Resize(x.shape[1:])(mean_pixels), min=0, max=255)

def toTensorNoScaling(x):
    return torch.from_numpy(np.array(x).transpose(2, 0, 1))

def toImageNoScaling(x):
  return Image.fromarray(np.uint8(np.array(x).transpose(1, 2, 0))).convert('RGB')

transform = transforms.Compose([
    # Transform to tensor without scaling
    transforms.Lambda(toTensorNoScaling),
    # Remove mean
    transforms.Lambda(meanSubstraction),
])

transform_inv = transforms.Compose([
    # Add mean
    transforms.Lambda(meanAddition),
    # To image
    transforms.Lambda(toImageNoScaling),
])

# Gram matrix
def gram_matrix(X):
  _, num_channels, height, width = X.shape
  X_vect = X.reshape((num_channels, height*width))
  return torch.matmul(X_vect, X_vect.T) / (num_channels * height * width)

def style_loss(Y_hat, Y):
    return torch.square(gram_matrix(Y_hat) - gram_matrix(Y)).mean()

def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y).mean()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg16 = VggNet()
    vgg16.load_state_dict(torch.load('data/model.pth'))
    vgg16.to(device)

    img = transform(Image.open('images/style/starry-night.jpg')).unsqueeze(0)
    img = img.to(device)
    style_layers = ['block1_1', 'block2_1', 'block3_1', 'block4_1', 'block5_1']
    ref_style_feature_map = vgg16.feature_maps(img, style_layers=style_layers, detach=True)

    img = transform(Image.open('images/content/eiffel-tower.jpg')).unsqueeze(0)
    img = img.to(device)
    content_layers=['block4_2']
    ref_content_feature_map = vgg16.feature_maps(img, content_layers=content_layers, detach=True)

    # Test style trasnfer
    gif = Animation()
    input = transform(Image.open('current.jpg')).unsqueeze(0).to(device) # 10 * torch.randn_like(img)
    input.requires_grad = True
    optimizer = torch.optim.Adam([input], lr=1.0) #torch.optim.LBFGS([input])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=2)
    current_im = transform_inv(input.detach().squeeze(0).cpu())
    current_im.save("current.jpg", "JPEG")

    style_weight = 1
    content_weight = 1e-3

    it        = 0
    prev_loss = 0.
    loss      = 1e3
    for ii in range(50000): #while abs(loss-prev_loss)>1e-2:
        it += 1
        prev_loss = loss
        optimizer.zero_grad()

        # Forward pass
        input.data = input.data.contiguous()
        content_feature_map, style_feature_map = vgg16.feature_maps(input, content_layers, style_layers)

        # Style loss
        s_loss = [style_weight * style_loss(style_feature_map[layer], ref_style_feature_map[layer]) for layer in style_layers]
        # Content loss
        c_loss = [content_weight * content_loss(content_feature_map[layer], ref_content_feature_map[layer]) for layer in content_layers]

        loss = sum(s_loss + c_loss)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        if ii%10==9:
            current_im = transform_inv(input.detach().squeeze(0).cpu())
            current_im.save("current.jpg", "JPEG")
            print(f"it {it} - Loss {loss}")
            server.global_state["iteration"] = it
            server.global_state["loss"]      = loss.item()

    print('Saving...')
    gif.save('style.gif')

    print('hola')
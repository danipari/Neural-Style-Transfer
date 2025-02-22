import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

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

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
    
    
# transpose is use to meet the shape of the tensor C x H x W rather than H x W x C
mean_pixels = np.float32(np.load('mean_pixels.npy').transpose(2, 0, 1))

def meanSubstraction(x):
    return x - mean_pixels

def meanAddition(x):
    return torch.clip(x + mean_pixels, min=0, max=255)

def toTensorNoScaling(x):
    return torch.from_numpy(np.array(x).transpose(2, 0, 1))

def toImageNoScaling(x):
  return Image.fromarray(np.uint8(np.array(x).transpose(1, 2, 0))).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
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

# From https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook
# a dict to store the activations
activation = {}
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output
    return hook

def get_feature_maps(input, model, layers):
    # Add hook to layers
    for layer in layers:
        n_block = int(layer[5])
        n_layer = int(layer[7])
        model.get_submodule(f"block{n_block}")[n_layer-1].register_forward_hook(getActivation(layer))

    # Run forward pass
    model(input)

    # Retrieve data
    out = dict()
    for layer in layers:
        out[layer] = activation[layer].detach().clone()

    return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg16 = VggNet()
    vgg16.load_state_dict(torch.load('model.pth'))
    vgg16.to(device)

    img = transform(Image.open('eiffel-tower.jpg')).unsqueeze(0)
    img = img.to(device)
    content_layer = ['block2_4']
    content_feature_map = get_feature_maps(img, vgg16, content_layer)

    img = transform(Image.open('starry-night.jpg')).unsqueeze(0)
    img = img.to(device)
    style_layers = ['block1_1', 'block2_1', 'block3_1', 'block4_1', 'block5_1']
    style_feature_map = get_feature_maps(img, vgg16, style_layers)

    # Test style trasnfer
    gif = Animation()
    input = 10 * torch.randn_like(img) # torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    input.requires_grad = True
    optimizer = torch.optim.LBFGS([input])
    gif.append(transform_inv(input.detach().squeeze(0).cpu()))

    it        = 0
    prev_loss = 0.
    loss      = 1e3
    for _ in range(100): #while abs(loss-prev_loss)>1e-2:
        it += 1
        prev_loss = loss
        def closure():
            optimizer.zero_grad()
            # Forward pass
            input.data = input.data.contiguous()
            vgg16(input)

            # Style loss
            style_loss = 0
            for layer in style_layers:
                style_loss += torch.square(gram_matrix(style_feature_map[layer])-gram_matrix(activation[layer])).mean()
            
            # Content loss
            content_loss = torch.square(content_feature_map[content_layer[0]]-activation[content_layer[0]]).mean()
            loss = 5e-3 * content_loss + 1 * style_loss
            loss.backward()
            return loss
        
        gif.append(transform_inv(input.detach().squeeze(0).cpu()))
        print(f"it {it} - Loss {loss}")
        loss = optimizer.step(closure)

    print('Saving...')
    gif.save('style.gif')

    print('hola')
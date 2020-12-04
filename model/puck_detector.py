import torch
import torch.nn.functional as F

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]

    # tip: visualize is_peak and heatmap side by side.
    is_peak = (heatmap >= max_cls).float()

    peaks = ((is_peak == 1).nonzero())
    
    result = []

    for peak in peaks:
      if (len(result) > max_det - 1):
        break
      x = peak[0]
      y = peak[1]

      if(heatmap[x][y] > min_score):
        result.append((heatmap[x][y].item(), y, x))

    return result



class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            x = x.to(device)
            #print((x.is_cuda))

            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L).to(device)
        self.classifier = torch.nn.Linear(c, n_output_channels).to(device)

    def forward(self, x):
        z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)).to(device)
        return self.classifier(z.mean(dim=[2, 3])).to(device)


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class PuckDetector(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1).to(device)

        def forward(self, x):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            return F.relu(self.c1(x)).to(device)
          
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=2, kernel_size=3, use_skip=True):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2).to(device))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2).to(device))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1).to(device)


    def forward(self, x):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        z = z.to(device)
        #print(self._modules["conv1"](z))
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        
        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        #print(z)
        # print("----")
        # print(z[0])
        # print(z.shape)
        # return torch.argmax(z)
        logit = self.classifier(z)
        return logit.mean(dim=[2,3])
        
        #logit = logit.squeeze(1)
        
        # print(logit.shape[])
        # return spatial_argmax(logit)


    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        # print(image.shape)
        #import pdb; pdb.set_trace()
        heatmap = self.forward(image)
        # print("Heatmap shape", heatmap.shape)
        
        listDetect1 = extract_peak(heatmap[0][0], max_det=30)
        listDetect2 = extract_peak(heatmap[0][1], max_det=30)
        listDetect3 = extract_peak(heatmap[0][2], max_det=30)

        widthHeight = (0,0)
        
        result1 = []
        result2 = []
        result3 = []
        
        for peak in listDetect1:
          result1.append(peak + widthHeight)
        
        for peak in listDetect2:
          result2.append(peak + widthHeight)
        
        for peak in listDetect3:
          result3.append(peak + widthHeight)
        
        return result1,result2,result3

       



# class PuckDetector(torch.nn.Module):
#     class UpBlock(torch.nn.Module):
#         def __init__(self, n_input, n_output, kernel_size=3, stride=2):
#             super().__init__()
#             self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
#                                       stride=stride, output_padding=1)

#         def forward(self, x):
#             return F.relu(self.c1(x))

#     class Block(torch.nn.Module):
#         def __init__(self, n_input, n_output, kernel_size=3, stride=2):
#             super().__init__()
#             self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
#                                       stride=stride, bias=False)
#             self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#             self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#             self.b1 = torch.nn.BatchNorm2d(n_output)
#             self.b2 = torch.nn.BatchNorm2d(n_output)
#             self.b3 = torch.nn.BatchNorm2d(n_output)
#             self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

#         def forward(self, x):
#             return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

#     def __init__(self, layers=[16, 32, 64, 128], n_output_channels=2, kernel_size=3, use_skip=True):
#         super().__init__()
#         self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
#         self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

#         c = 3
#         self.use_skip = use_skip
#         self.n_conv = len(layers)
#         skip_layer_size = [3] + layers[:-1]
#         for i, l in enumerate(layers):
#             self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
#             c = l
#         for i, l in list(enumerate(layers))[::-1]:
#             self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
#             c = l
#             if self.use_skip:
#                 c += skip_layer_size[i]

#         #self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)
#         self.classifier = torch.nn.Linear(c, n_output_channels)

#     def forward(self, x):
#         z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
#         up_activation = []
#         for i in range(self.n_conv):
#             # Add all the information required for skip connections
#             up_activation.append(z)
#             z = self._modules['conv%d'%i](z)

#         for i in reversed(range(self.n_conv)):
#             z = self._modules['upconv%d'%i](z)
#             # Fix the padding
#             z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
#             # Add the skip connection
#             if self.use_skip:
#                 z = torch.cat([z, up_activation[i]], dim=1)

#         z = z.mean(dim=[2,3])

#         return self.classifier(z)

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, PuckDetector):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'puck_det.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = PuckDetector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'puck_det.th'), map_location='cpu'))
    return r
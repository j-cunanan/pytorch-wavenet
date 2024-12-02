import os
import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from wavenet_modules import dilate, DilatedQueue, constant_pad_1d
from audio_data import mu_law_expansion

class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N * L_{out}, C_{out})`
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                  out_channels=residual_channels,
                                  kernel_size=1,
                                  bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                      num_channels=residual_channels,
                                                      dilation=new_dilation,
                                                      dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=kernel_size,
                                               bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=1,
                                                   bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                               out_channels=skip_channels,
                                               kernel_size=1,
                                               bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                   out_channels=end_channels,
                                   kernel_size=1,
                                   bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                   out_channels=classes,
                                   kernel_size=1,
                                   bias=True)

        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):
        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            (dilation, init_dilation) = self.dilations[i]
            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                         dilation=dilation)
        x = x.unsqueeze(0)
        return x

    def forward(self, input):
        x = self.wavenet(input, dilation_func=self.wavenet_dilate)
        
        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(n * l, c)
        return x

    def generate(self, num_samples, first_samples=None, temperature=1.):
        self.eval()
        with torch.no_grad():
            if first_samples is None:
                first_samples = torch.zeros(1, dtype=torch.float32, device=self.start_conv.weight.device)
            generated = first_samples

            num_pad = self.receptive_field - generated.size(0)
            if num_pad > 0:
                generated = constant_pad_1d(generated, self.receptive_field, pad_start=True)

            for i in range(num_samples):
                input = torch.zeros(1, self.classes, self.receptive_field, 
                                  device=self.start_conv.weight.device)
                input = input.scatter_(1, generated[-self.receptive_field:].long().view(1, -1, self.receptive_field), 1.)

                x = self.wavenet(input, dilation_func=self.wavenet_dilate)[:, :, -1].squeeze()

                if temperature > 0:
                    x /= temperature
                    prob = F.softmax(x, dim=0)
                    prob = prob.cpu().numpy()
                    x = np.random.choice(self.classes, p=prob)
                    x = torch.tensor([x], device=self.start_conv.weight.device)
                else:
                    x = torch.argmax(x, dim=0).float()

                generated = torch.cat((generated, x))

            generated = (generated / self.classes) * 2. - 1
            mu_gen = mu_law_expansion(generated, self.classes)

        self.train()
        return mu_gen

    def generate_fast(self, num_samples, first_samples=None, temperature=1., regularize=0.,
                     progress_callback=None, progress_interval=100):
        self.eval()
        with torch.no_grad():
            if first_samples is None:
                first_samples = torch.zeros(1, dtype=torch.long, device=self.start_conv.weight.device) + (self.classes // 2)

            # reset queues
            for queue in self.dilated_queues:
                queue.reset()

            num_given_samples = first_samples.size(0)
            total_samples = num_given_samples + num_samples

            input = torch.zeros(1, self.classes, 1, device=self.start_conv.weight.device)
            input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

            # fill queues with given samples
            for i in range(num_given_samples - 1):
                x = self.wavenet(input, dilation_func=self.queue_dilate)
                input.zero_()
                input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

                if i % progress_interval == 0 and progress_callback is not None:
                    progress_callback(i, total_samples)

            # generate new samples
            generated = np.array([])
            regularizer = torch.arange(self.classes, device=self.start_conv.weight.device)
            regularizer = torch.pow(regularizer - self.classes / 2., 2) * regularize

            tic = time.time()
            for i in range(num_samples):
                x = self.wavenet(input, dilation_func=self.queue_dilate).squeeze()
                x = x - regularizer

                if temperature > 0:
                    # sample from softmax distribution
                    x /= temperature
                    prob = F.softmax(x, dim=0)
                    prob = prob.cpu().numpy()
                    x = np.random.choice(self.classes, p=prob)
                    x = np.array([x])
                else:
                    # convert to sample value
                    x = torch.argmax(x, dim=0).cpu().numpy()

                o = (x / self.classes) * 2. - 1
                generated = np.append(generated, o)

                # set new input
                x = torch.from_numpy(x).to(device=self.start_conv.weight.device, dtype=torch.long)
                input.zero_()
                input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

                if (i + 1) == 100:
                    toc = time.time()
                    print(f"one generating step takes approximately {(toc - tic) * 0.01:.3f} seconds")

                if (i + num_given_samples) % progress_interval == 0 and progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def to(self, device):
      """Override the to() method to properly move queue data."""
      # First move the model using parent's to() method
      super().to(device)
      # Then move all dilated queues
      for queue in self.dilated_queues:
          queue.to(device)
      return self

def load_latest_model_from(location, device='cpu'):
    files = [os.path.join(location, f) for f in os.listdir(location)]
    if not files:
        raise FileNotFoundError(f"No model files found in {location}")
    
    newest_file = max(files, key=os.path.getctime)
    print(f"Loading model from {newest_file}")

    # First create a new model instance
    model = WaveNetModel(
        layers=10,  # You might want to make these parameters configurable
        blocks=3,
        dilation_channels=32,
        residual_channels=32,
        skip_channels=1024,
        end_channels=512,
        output_length=16,
        bias=True
    )
    
    # Load the state dict
    if device == 'cuda' and torch.cuda.is_available():
        state_dict = torch.load(newest_file)
        model.cuda()
    else:
        state_dict = torch.load(newest_file, map_location='cpu')
        model.cpu()
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    return model

def load_to_cpu(path):
    # Create a new model instance
    model = WaveNetModel(
        layers=10,
        blocks=3,
        dilation_channels=32,
        residual_channels=32,
        skip_channels=1024,
        end_channels=512,
        output_length=16,
        bias=True
    )
    
    # Load state dict to CPU
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model
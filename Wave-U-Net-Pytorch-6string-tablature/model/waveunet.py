import torch
import torch.nn as nn

import torch.nn.functional as F

from model.crop import centre_crop
from model.resample import Resample1d
from model.conv import ConvLayer
import torch.nn.init as init # TabCNN
import librosa
import numpy as np
# import essentia.standard as es



class TabCNN(nn.Module):
    def __init__(self, num_strings=6, num_classes=21, input_shape=(192, 9, 1)):
        super(TabCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(5952, 128)  # now we use the correct number here
        self.fc2 = nn.Linear(128, num_classes * num_strings)

        self.num_strings = num_strings
        self.num_classes = num_classes

        # Initialize weights with Xavier/Glorot Uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        encoding = x.clone()
      
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = x.view(-1, self.num_strings, self.num_classes)
        return x, encoding.unsqueeze(0)  


class WaveUNetHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_strings):
        super(WaveUNetHead, self).__init__()
        self.wfc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.wfc2 = nn.Linear(hidden_dim, num_classes * num_strings)

    def forward(self, x):
        x = torch.relu(self.wfc1(x))
        x = self.dropout(x)
        x = self.wfc2(x)
        x = x.view(-1, 6, 21).unsqueeze(0)  # Reshape to (1, T, 6, 21)
        x = x.permute(0, 3, 2, 1)  # Permute to (1, 21, 6, T)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)
        # depth is by default set to 1

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))

        return combined, upsampled

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size


class TabDownsamplingBlock(nn.Module): # __gbastas_
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(TabDownsamplingBlock, self).__init__()
        assert(stride > 1)
        # depth is by default set to 1

        self.kernel_size = kernel_size
        self.stride = stride
        # self.Tab = Tab

        # CONV 1
        self.pre_shortcut_conv = ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)
        self.post_shortcut_conv = ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x, shortcut):
        # PREPARING FOR DOWNSAMPLING
        out = x
        out = self.pre_shortcut_conv(out)

        # DOWNSAMPLING
        try:
            combined = centre_crop(shortcut[:,:,:], out)
        except Exception:
            combined = centre_crop(shortcut[:,:,:-1], out)

        # Combine high- and low-level features
        out = self.post_shortcut_conv(torch.cat([combined, centre_crop(out, combined)], dim=1))
        out = self.downconv(out)       

        return out, combined



class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res, padding=0, Tab=False):#tab_version=None):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        # depth is by default set to 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.Tab = Tab


        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type, padding=padding)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type, padding=padding) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type, padding=padding)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type, padding=padding) for _ in range(depth - 1)])


        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type, padding=padding)


    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        if self.Tab:
            out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size, conv_type, res, separate=False, depth=1, strides=2, tab_version='4up3down'):
        super(Waveunet, self).__init__()
        """
        num_inputs is args.channels i.e. 1
        num_channels is [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                        # [args.features*2**i for i in range(0, args.levels)] where args.feaures i.e. 32
        num_outputs is args.channels i.e. 1
        kernel_size is args.kernel_size i.e. 5
        target_output_size = int(args.output_size * args.sr)
        """        

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate
        self.num_channels = num_channels
        self.conv_type = conv_type
        self.res = res
        self.tab_version = tab_version
        
        
        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = nn.ModuleDict()

        self.model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in self.model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size) # [gb] used for data loading, see data/dataset.py

        self.tab_branches_added = False  # __gabastas__ Indicator whether tab branches are added

    def add_tab_branches(self):  # __gbastas__
        if 'TabCNN' in self.tab_version:
            self.tab_cnn = TabCNN()
            self.tab_cnn.load_state_dict(torch.load("./model/tabcnn-mic/senvaityte/model.pth"))

            self.waveunet_head = WaveUNetHead(input_dim=7488, hidden_dim=128, num_classes=21, num_strings=6) # new

        for instrument in self.instruments:

            if self.tab_version=='2up2down' or self.tab_version=='2up2down-TabCNN':
                padding = (self.kernel_size - 1) // 2 # (=2)
                self.waveunets[instrument].tab_downsampling_block1 = DownsamplingBlock(self.num_channels[-6], self.num_channels[-6], self.num_channels[-5], self.kernel_size, self.strides, 1, self.conv_type, 'non-fixed', padding=padding)
                self.waveunets[instrument].tab_downsampling_block2 = DownsamplingBlock(self.num_channels[-5], self.num_channels[-5], self.num_channels[-4], self.kernel_size, self.strides, 1, self.conv_type, 'non-fixed', padding=padding)
                self.waveunets[instrument].tab_downsampling_block3 = DownsamplingBlock(self.num_channels[-4], self.num_channels[-4], self.num_channels[-4], self.kernel_size, self.strides, 1, self.conv_type, 'non-fixed', padding=padding)
                self.waveunets[instrument].tab_downsampling_block4 = DownsamplingBlock(self.num_channels[-4], self.num_channels[-4], self.num_channels[-4], self.kernel_size, self.strides, 1, self.conv_type, 'non-fixed', padding=padding)
                self.waveunets[instrument].tab_downsampling_block5 = DownsamplingBlock(self.num_channels[-4], self.num_channels[-4], self.num_channels[-4], self.kernel_size, self.strides, 1, self.conv_type, 'non-fixed', padding=padding)

            self.waveunets[instrument].tab_fc1 = nn.Linear(self.num_channels[-4], self.num_channels[-4])
            self.waveunets[instrument].tab_fc2 = nn.Linear(self.num_channels[-4], 21)

        self.tab_branches_added = True

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        # [gb] These are used for data loading, see data/dataset.py
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module, tabcnn_pred=None):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        upsampled_short = []
        out = x
        tab_out = None
        
        # DOWNSAMPLING BLOCKS
        for idx, block in enumerate(module.downsampling_blocks):
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out, short = block(out, shortcuts[-1 - idx])

            if idx==4: # i.e 5 upsampling layers
                tab_out = out.clone()
            upsampled_short.append(short)
        tab_enc = None
        if self.tab_branches_added:
            if self.tab_version=='2up2down' or self.tab_version=='2up2down-TabCNN':
                tab_out, _ = module.tab_downsampling_block1(tab_out)
                tab_out, _ = module.tab_downsampling_block2(tab_out)
                tab_out, _ = module.tab_downsampling_block3(tab_out)
                tab_out, _ = module.tab_downsampling_block4(tab_out)
                tab_out, _ = module.tab_downsampling_block5(tab_out)
                if 'TabCNN' in self.tab_version:
                    ################
                    tab_enc = tab_out.clone()
                    ################

            tab_out = tab_out.transpose(1, 2)  # Prepare for dense layer (N, T, C)
     
            tab_out = torch.relu(module.tab_fc1(tab_out))
            tab_out = module.tab_fc2(tab_out)
            tab_out = tab_out.transpose(1, 2)  # Return to (N, C, T) format
        
            # Resample
            indices = np.linspace(0, tab_out.shape[0] - 1, 87, endpoint=True).astype(int)
            tab_out = tab_out[:, :, indices]            


        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)

        return out, tab_out, tab_enc

    def forward(self, x, x_cqt=None, inst=None):
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size)  # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size
        assert(inst is None)
        tab_list = []
        tab_enc_list = []
        tab_outs = {}
        aggr_tab_pred = 90



        if self.tab_version != 'TabCNN':
            for inst in self.model_list:
                out, tab_out, tab_enc = self.forward_module(x, self.waveunets[inst])

                tab_enc_list.append(tab_enc)
                tab_outs[inst] = {"output": out, "tab_pred": tab_out}  # tab_pred changes value further on for TabCNN version

        if 'TabCNN' in self.tab_version:
            ############ TabCNN ############
            x_cqt_tensor = x_cqt  # Shape: (1, 192, T)
            overlapping_windows = create_overlapping_windows(x_cqt_tensor, window_size=9, step_size=1)  # Break it to 9 frame batches!
            tabcnn_pred, tabcnn_enc = self.tab_cnn(overlapping_windows)  # Assuming overlapping_windows has shape (N, 1, 192, 9), where N is the number of windows (N=342)
            ############ TabCNN ############        

            if self.tab_version == '2up2down-TabCNN':
                concatenated_tab_enc = torch.cat(tab_enc_list, dim=1)
                concatenated_tab_enc = concatenated_tab_enc.transpose(1, 2)  # Prepare for dense layer (N, T, C), e.g. [1, 342, 3072]             
                aggr_tab_pred = torch.cat((tabcnn_enc, concatenated_tab_enc), dim=2) # 1, 87, 5952 and 1, 87, 1536

                # Output Head for WaveUNet-TabCNN
                aggr_tab_pred = self.waveunet_head(aggr_tab_pred)
                

            elif self.tab_version == 'TabCNN':
                aggr_tab_pred = tabcnn_pred.unsqueeze(0)
                aggr_tab_pred = aggr_tab_pred.permute(0, 3, 2, 1)

        return tab_outs, aggr_tab_pred



        

def create_overlapping_windows(input_tensor, window_size=9, step_size=1):
    """
    Create overlapping windows from the input tensor.
    
    Parameters:
    - input_tensor: Tensor of shape (1, 192, T)
    - window_size: Number of frames in each window
    - step_size: Step size between consecutive windows
    
    Returns:
    - A tensor of shape (N, 1, 192, 9) where N is the number of windows
    """
    _, _, T_padded = input_tensor.shape  # New T after padding
    num_windows = (T_padded - window_size) // step_size + 1
    windows = [input_tensor[..., i:i+window_size] for i in range(0, num_windows*step_size, step_size)]

    # Stack the windows along a new dimension
    windows_tensor = torch.stack(windows, dim=0)

    return windows_tensor
        
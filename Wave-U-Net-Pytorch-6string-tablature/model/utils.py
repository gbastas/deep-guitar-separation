import os
import torch
import torch.nn as nn
import numpy as np

def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
    }, path)


def load_model(model, optimizer, path, cuda, strict=True):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
        
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed, strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state

def avg_acc_func(y_true, y_pred): # Tab-CNN
    y_true = torch.argmax(y_true, dim=-1)
    y_pred = torch.argmax(y_pred, dim=-1)   
    # print(y_true.size(), y_pred.size())
    accuracy = torch.mean(torch.eq(y_true, y_pred).float())
    return accuracy


class AggrTabLoss(nn.Module):
    def __init__(self, num_strings=6, close_weight=0.5):
        super(AggrTabLoss, self).__init__()
        self.num_strings = num_strings
        self.close_weight = close_weight

    def forward(self, output, target):
        loss = 0
        criterion = nn.CrossEntropyLoss()
        for i in range(self.num_strings):
            # N x 21 x 6 x T ,  N x T x 6
            loss += criterion(output[:, :, i, :], target[:, :, i])  # N X 21 X T, N x T

        return loss

def compute_loss(model, inputs, targets, tab_labels, cqt, criterion, tab_criterion, compute_grad=False, task='separation', tab_version='4up3down'):
    '''
    Computes gradients of model with given inputs and targets and loss function.
    Optionally backpropagates to compute gradients for weights.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param inputs: Input mixture
    :param targets: Target sources
    :param criterion: Loss function to use (L1, L2, ..)
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batchn
    '''
    all_sep_outputs = {}
    y_preds = []
    # softmax_layer = SoftmaxByString(6)

    if model.separate:
        avg_sep_loss = 0.0
        num_sources = 0
        avg_tab_loss = 0.0
        avg_tab_acc = 0.0
        num_sources = 0

        output, aggr_tab_out = model(inputs, cqt, inst=None) # aggr_tab_out: N x 21 x 6 x T    


        for inst in model.instruments:
            # output = model(inputs, cqt, inst) # NOTE.EEEEEEEEEEEE

            if task in ['separation', 'multitask']:
                sep_loss = criterion(output[inst]['output'], targets[inst])
                if compute_grad:
                    sep_loss.backward(retain_graph=True)
                # print('sep_loss', sep_loss)
                avg_sep_loss += sep_loss.item()
                all_sep_outputs[inst] = output[inst]['output'].detach().cpu().clone()


            if task in ['tablature', 'multitask']:
                class_indices = tab_labels[inst].max(dim=1)[1]
                if 'TabCNN' not in tab_version and output[inst]['tab_pred'] is not None:
                    tab_loss = tab_criterion(output[inst]['tab_pred'], class_indices)     
                    # y_preds += [output[inst]['tab_pred']]
                    if compute_grad:
                        # if 'TabCNN' in tab_version:
                        #     tab_loss.backward(retain_graph=True) 
                        # else:
                        tab_loss.backward()
                        # print('tab_loss', tab_loss)

                    avg_tab_loss += tab_loss.item()
                    # soft_outputs = softmax_layer(outputs)
   

        ####################################################
        if 'TabCNN' in tab_version or tab_version == '2up2down-TabCNN':
            aggr_tab_labels = []
            for inst in model.instruments:
                aggr_tab_labels.append(tab_labels[inst])

            aggr_tab_labels = torch.cat(aggr_tab_labels, dim=0)   # (6, 21, T) 
            aggr_tab_labels = aggr_tab_labels.unsqueeze(0) # (1, 6, 21, T) 
            aggr_tab_labels = aggr_tab_labels.transpose(1,2) # (1, 21, 6, T) 
            aggr_tab_labels = aggr_tab_labels.max(dim=1)[1] # (1, 6, T)
            aggr_tab_labels = aggr_tab_labels.transpose(1,2) # (1, T, 6)

            aggr_tab_criterion = AggrTabLoss()


            aggr_tab_loss = aggr_tab_criterion(aggr_tab_out, aggr_tab_labels)

            if compute_grad:
                aggr_tab_loss.backward()

            avg_tab_loss += aggr_tab_loss.item()
            # avg_tab_loss /=2 # NOTE

            # # y_preds = np.array(aggr_tab_out)    # (6, 21, 78760)
            # y_preds = aggr_tab_out.detach().cpu().numpy()
            # y_preds = y_preds.squeeze(0)    # (21, 6, 78760)
            # # print('aggr_tab_out', y_preds.shape)
            # y_preds = np.transpose(y_preds, (2, 1, 0)) # (78760, 6, 21)
            # print_tablature(y_preds)                  
        ####################################################

        # Calculate the average losses
        num_sources=6
        avg_sep_loss /= float(num_sources)
        avg_tab_loss /= float(num_sources)
        avg_tab_acc /= float(num_sources)
    else:
        print("This doesn't work --separate", model.separate)
        exit(0)

    # print('avg_loss', avg_loss, 'avg_tab_loss', avg_tab_loss)
    # print('avg_loss', None, 'avg_tab_loss', avg_tab_loss)
    # print('avg_tab_acc', avg_tab_acc)

    return all_sep_outputs, avg_sep_loss, avg_tab_loss, avg_tab_acc


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def print_tablature(tab_array_N_6_21, max_chars_per_line=140):
    # Initialize the tablature lines
    tab_array_N_6 = np.array(list(map(tab2bin,tab_array_N_6_21)))
    maxnum =100
    tab = tab_array_N_6
    tab_lines = ['-' * (len(tab[:maxnum,:]) * 3) for _ in range(6)]

    # Fill in the tablature lines with the fret numbers
    for i in range(len(tab[:maxnum,:])):
        for j in range(6):

            if tab[i, j] != -1:
                tab_lines[j]= tab_lines[j][:i*3] +'-' + str(int(tab[i, j])) + '-' + tab_lines[j][i*3+3:]
    for j in range(6):
        print(tab_lines[j])
        
    print()

        
def tab2bin(tab):
    tab_arr = np.zeros(6)
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed 
        # if fret_class > 0:
        fret_num = fret_class - 1
        tab_arr[string_num] = fret_num # __gbastas__
    return tab_arr                    
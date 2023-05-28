import torch
from torch_utils import persistence
import numpy as np
from torch.autograd import grad

class Sine(torch.nn.Module):
  '''  See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for the
  discussion of factor 30
  '''

  def __init__(self, w=30.0):
    super().__init__()
    self.w = w

  def forward(self, input):
    return torch.sin(self.w * input)

  def extra_repr(self):
    return 'w={}'.format(self.w)

class Embedder(object):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        embed_fns = []
        input_dims = self.kwargs['input_dims']
        output_dims = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            output_dims += input_dims
        
        max_freq = self.kwargs['max_freq_log2']
        num_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = torch.pi * 2.**torch.linspace(0., max_freq, steps=num_freqs)
        else:
            freq_bands = torch.pi * torch.linspace(2.**0., 2.**max_freq, steps=num_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                output_dims += input_dims
        
        self._embed_fns = embed_fns
        self._output_dims = output_dims

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self._embed_fns], dim=-1)

    def output_dims(self):
        return self._output_dims


def get_embedder(multires, no_encoding=False, include_input=True, include_tanh=False):
    if no_encoding:
        include_input = True
        multires = 0
    
    if include_tanh:
        periodic_fns = [torch.sin, torch.cos, lambda x: torch.tanh(x/4.)]
    else:
        periodic_fns = [torch.sin, torch.cos]

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': periodic_fns,
    }

    return Embedder(embed_kwargs)

class IGRModel(torch.nn.Module):
    def __init__(self, 
        intermediate_dims,
        skip_connection_layers,
        bbox,
        feature_dim=0,
        output_dim=1, 
        geometric_init=True,
        radius_init=1,
        activation_func='swish',
        template_sdf=None):
        super(IGRModel, self).__init__()

        self._skip_connection_layers = skip_connection_layers
        self._feature_dim = feature_dim
        self._output_dim = output_dim
        self._embedding_dim = 9
        dims = [self._embedding_dim+feature_dim] + intermediate_dims + [output_dim]

        self._num_layers = len(dims)

        bbmin, bbmax = bbox
        self._bbox_center = 0.5 * (bbmin + bbmax)
        self._edge = 0.5 * (bbmax - bbmin)
        self._inv_edge_scaling = 2.0/(bbmax - bbmin)

        self._template_sdf = template_sdf

        if activation_func == 'swish':
            self._activation = torch.nn.SiLU()
        elif activation_func == 'softplus':
            self._activation = torch.nn.Softplus()
        elif activation_func == 'sine':
            self._activation = Sine()
        else:
            self._activation = torch.nn.ReLU()

        for layer in range(0, self._num_layers - 1):
            in_dim = dims[layer]
            out_dim = dims[layer+1]
            if layer in self._skip_connection_layers:
                in_dim += self._embedding_dim + feature_dim
            fc_layer = torch.nn.Linear(in_dim, dims[layer+1])

            if geometric_init:
                if layer == self._num_layers - 2:
                    torch.nn.init.normal_(fc_layer.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(fc_layer.bias, -radius_init)
                else:
                    torch.nn.init.constant_(fc_layer.bias, 0.0)

                    torch.nn.init.normal_(fc_layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
            setattr(self, "fc" + str(layer), fc_layer)

    def run_mlp(self, input_points, inputs, mask=None):
        
        if mask is not None:
            input_points = input_points[mask]
        embed_inputs = torch.cat([input_points, torch.sin(torch.pi*input_points), torch.cos(torch.pi*input_points)], dim=-1)
        if 'features' in inputs:
            features = inputs['features'][mask]
            embed_inputs = torch.cat([embed_inputs, features], dim=-1)

        x = embed_inputs
        mlp_features = None
        for layer in range(self._num_layers - 1):

            fc_layer = getattr(self, "fc" + str(layer))

            if layer in self._skip_connection_layers:
                x = torch.cat([x, embed_inputs], dim = -1) / np.sqrt(2.)
            
            x = fc_layer(x)

            if layer < self._num_layers - 2:
                x = self._activation(x)
                mlp_features = x

        if self._template_sdf:
            x[..., -3:] = x[..., -3:] * self._edge + self._bbox_center

        return x, mlp_features

    def run_approx(self, input_points):
        if self._output_dim == 3:
            return input_points
        dist = torch.norm(input_points, dim=-1, keepdim=True) - 0.045
        if self._output_dim == 1:
            return dist
        else:
            return torch.cat([dist, input_points], dim=-1)

    def forward(self, inputs):
        input_points = inputs['points']

        device = input_points.device
        self._bbox_center = self._bbox_center.to(device)
        self._inv_edge_scaling = self._inv_edge_scaling.to(device)
        self._edge = self._edge.to(device)

        # transform the points into the unit box of [-1, 1]
        scaled_input_points = (input_points - self._bbox_center) * self._inv_edge_scaling

        inside_bbox = torch.sum((scaled_input_points <= 1).float() + (scaled_input_points >= -1).float(), dim=-1) == 6
        return_distance_only = inputs['return_distance_only']
        return_warping_only = inputs['return_warping_only']

        x = self.run_approx(input_points)
        x[inside_bbox], mlp_features = self.run_mlp(scaled_input_points, inputs, inside_bbox)
        
        if return_warping_only:
            return x, inside_bbox, mlp_features
            
        if self._template_sdf:
            query = {
                'points': x[..., -3:],
                'return_warping_only': True,
                'return_distance_only': True,
            }
            dist = self._template_sdf(query)
        else:
            dist = x[..., :1]

        if return_distance_only:
            return dist
        else:
            if self._template_sdf:
                return torch.cat([dist, x], dim=-1)
            else:
                return x
    
    def eval_and_gradient(self, input_kwargs):
        input_points = input_kwargs['points']
        input_points.requires_grad_(True)
        outputs = self.forward(input_kwargs)
        dists = outputs[...,:1]
        d_points = torch.ones_like(dists, requires_grad=False, device=dists.device)
        points_grad = grad(
            outputs=dists,
            inputs=input_points,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0][:, -3:]
        
        return outputs, points_grad
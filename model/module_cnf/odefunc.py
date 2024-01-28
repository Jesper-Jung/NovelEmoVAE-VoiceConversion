import copy
import torch
import torch.nn as nn
from . import diffeq_layers

__all__ = ["ODEnet", "ODEfunc"]

def divergence_approx(f, y, e=None):
    """ Compute vector-Jacobian product with automatic differentiation,
        and unbiased estimate.

        #=== INPUT

        - f: the function which we want to calculate gradient
        - y: input vector of the gradient.
        - e: output_grad, here it is given by the noise.

    """
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        # print("RequiresGrad:f=%s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt=%d"
        #       % (f.requires_grad, y.requires_grad, e_dzdx.requires_grad,
        #          e.requires_grad, e_dzdx_e.requires_grad, cnt))
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, \
        "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
        % (
        f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    return approx_tr_dzdx


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, dim_spk, dim_emo, layer_type="concat", nonlinearity="softplus"):
        super(ODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "squash": diffeq_layers.SquashLinear,
            "scale": diffeq_layers.ScaleLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "spkemo": diffeq_layers.SpkEmoLinear,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "concatscale": diffeq_layers.ConcatScaleLinear,
        }[layer_type]

        # build models and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out in (hidden_dims + (input_shape[0],)):

            layer_kwargs = {}
            layer = base_layer(hidden_shape[0], dim_out, dim_spk, dim_emo, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, y, t, spk_emb, emo_emb):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(dx, t, spk_emb, emo_emb)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        """
        #=== INPUT
        • t         || number?
            It is just a time variable.
            
        • states    ||  [(batch, dim_noise), (batch, 1), (batch, dim_spk), (batch, dim_emo)]
            Here, len(states) == 4, this flow is conditional.
            Sequentially, contents in the states is input, logpx, and condition vector.
        """
        
        y = states[0]
        batch_size = y.size(0)
        
        t = torch.ones(batch_size, 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            
            #=== conditional CNF
            if len(states) == 4:  
                spk_emb, emo_emb = states[2], states[3]

                dy = self.diffeq(y, t, spk_emb, emo_emb)
                divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)         # (batch, 1)

                return dy, -divergence, \
                    torch.zeros_like(spk_emb).requires_grad_(True), torch.zeros_like(emo_emb).requires_grad_(True) 
                    # (batch, noise_dim), (batch, 1), (batch, dim_spk), (batch, dim_emo)
            
            
            #=== unconditional CNF
            elif len(states) == 2:  
                dy = self.diffeq(t, y)  
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
                
                return dy, -divergence
            
            #=== excerpt
            else:
                assert 0, "`len(states)` should be 2 or 4"




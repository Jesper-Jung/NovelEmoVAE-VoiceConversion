import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

from model.speakerEmbedder import kwarg_SPEAKER
device = 'cuda' if torch.cuda.is_available() else 'cpu'

__all__ = ["CNF", "SequentialFlow"]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list, config):
        super(SequentialFlow, self).__init__()
        
        self.n_emo = config['Model']['n_emo']
        
        self.use_one_hot = config['Model']['use_one_hot']
        self.use_spk_linear = config['Model']['use_spk_linear']
        
        if self.use_one_hot:
            dim_emo = self.n_emo
        else:
            dim_emo = config['Model']['Style_Prior']['dim_emo']
            self.emo_embed = nn.Embedding(self.n_emo, dim_emo)
            
        dim_spk = config['Model']['Style_Prior']['dim_spk']
        
        self.chain = nn.ModuleList(layer_list)
        if self.use_spk_linear:
            self.spk_linear = nn.Sequential(
                nn.Linear(kwarg_SPEAKER['nOut'], dim_spk),
                nn.GELU(),
                nn.Linear(dim_spk, dim_spk),
                nn.GELU(),
                nn.Linear(dim_spk, dim_spk),
                nn.GELU(),
                nn.Linear(dim_spk, dim_spk)
            )

    def forward(self, x, spk_emb, emo_id, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))
        
        if self.use_one_hot:
            emo_emb = F.one_hot(emo_id.to(torch.int64), num_classes=self.n_emo).float()
        else:
            emo_emb = self.emo_embed(emo_id)        # (B, dim_emo)
            
        if self.use_spk_linear:
            spk_emb = self.spk_linear(spk_emb)          # (B, dim_spk)

        if logpx is None:
            for i in inds:
                # print(x.shape)
                x = self.chain[i](x, spk_emb, emo_emb, logpx, integration_times, reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, spk_emb, emo_emb, logpx, integration_times, reverse)
            return x, logpx


class CNF(nn.Module):
    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super(CNF, self).__init__()
        
        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        
        self.use_adjoint = use_adjoint
        self.conditional = conditional
        
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
            print("Training T :", self.T)

        self.odefunc = odefunc
        self.solver = solver
        self.solver_options = {}
        
        self.atol = atol
        self.rtol = rtol
        
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol


    def forward(self, x, spk_emb, emo_emb, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx

        if self.conditional:
            assert spk_emb is not None
            states = (x, _logpx, spk_emb, emo_emb)
            # atol = [self.atol] * 4
            # rtol = [self.rtol] * 4
        else:
            states = (x, _logpx)
            # atol = [self.atol] * 2
            # rtol = [self.rtol] * 2
            
        atol = self.atol
        rtol = self.rtol
        

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
               # print("integration times:", integration_times)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(x)

        if reverse:
            integration_times = _flip(integration_times, 0)
            
            
        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        
        # solve odeint, which outputs [T, Batch, D]
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        
        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
            
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)


        z_t, logpz_t = state_t[:2]

        if logpx is not None:
            return z_t, logpz_t         # (batch, noise_dim), (batch, 1)
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
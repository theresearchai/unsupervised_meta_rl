import torch
import numpy as np

from garage.torch.modules import MLPModule
import garage.torch.utils as tu

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20

class GMM():
    def __init__(self,
                 K,
                 Dx,
                 mlp_input_dim=None,
                 hidden_layer_sizes=(124, 124),
                 reg=0.001,
                 reparameterize=True,
                 ):
        self._reg = reg
        self._reparameterize = reparameterize

        self._Dx = Dx
        self._K = K
        if mlp_input_dim == None:
            self._w_and_mu_logsig_t = torch.distributions.normal.Normal(0, 0.1)
            self._use_mlp = False
        else:
            self._w_and_mu_logsig_t = MLPModule(input_dim=mlp_input_dim,
                                                output_dim=K * (2 * Dx + 1),
                                                hidden_sizes=hidden_layer_sizes)
            self._use_mlp = True

    def get_p_xz_params(self, input):
        K = self._K
        Dx = self._Dx
        output = self._w_and_mu_logsig_t(input).reshape(-1, K, 2*Dx+1)
        log_w_t = output[..., 0]
        mu_t = output[..., 1:1+Dx]
        log_sig_t = output[..., 1+Dx:]
        log_sig_t = torch.clamp(log_sig_t, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)

        return log_w_t, mu_t, log_sig_t

    def get_p_params(self, input):
        log_ws_t, xz_mus_t, xz_log_sigs_t = self.get_p_xz_params(input)
        # (N x K), (N x K x Dx), (N x K x Dx)
        N = log_ws_t.shape[0]
        xz_sigs_t = torch.exp(xz_log_sigs_t)

        # Sample the latent code
        z_t = torch.multinomial(torch.exp(log_ws_t), num_samples=1)  # N*1

        # Choose mixture component corresponding to the latent
        mask_t = torch.eye(self._K)[z_t[:, 0]].to(tu.global_device())
        mask_t = mask_t.ge(1) # turn into boolean
        xz_mu_t = torch.masked_select(xz_mus_t, mask_t)
        xz_sig_t = torch.masked_select(xz_sigs_t, mask_t)

        # Sample x
        x_t = xz_mu_t + xz_sig_t * torch.normal(mean=torch.zeros((N, self._Dx)).to(tu.global_device()),
                                                std=1.0)

        if not self._reparameterize:
            x_t = x_t.detach().cpu().numpy()

        # log p(x|z)
        log_p_xz_t = self._create_log_gaussian(xz_mus_t, xz_log_sigs_t, x_t[:, None, :])
        # N*K

        # log p(x)
        log_p_x_t = torch.logsumexp(log_p_xz_t + log_ws_t, dim=1)
        log_p_x_t -= torch.logsumexp(log_ws_t, dim=1)

        reg_loss_t = 0
        reg_loss_t += self._reg * 0.5 * torch.mean(xz_log_sigs_t ** 2)
        reg_loss_t += self._reg * 0.5 * torch.mean(xz_mus_t ** 2)

        return log_p_x_t, reg_loss_t, x_t, log_ws_t, xz_mus_t, xz_log_sigs_t

    @staticmethod
    def _create_log_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)
        quadratic = -0.5 * torch.sum(normalized_dist_t ** 2, dim=-1)

        log_z = torch.sum(log_sig_t, dim=-1)
        D_t = torch.tensor(mu_t.shape[-1]*1.0)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p

    @property
    def parameters(self):
        return self._w_and_mu_logsig_t.parameters if self._use_mlp is True else None

    @property
    def networks(self):
        return self._w_and_mu_logsig_t if self._use_mlp is True else None






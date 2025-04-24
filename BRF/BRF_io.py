import torch
import torch.nn as nn



class BRF_cell(nn.Module):
    def __init__(
            self,
            W, # adjacency matric for gap coupling
            b, # beta values for simulation
            omega, 
            q_decay=0.95, # decay for refractory period
            b_eff_scale=2.0, # scale q by this so post spike dampening/soft reset is more effective
            # effective dampening VERY WEAK unless scaled, example: test 1.0
            theta=1.0, # threshold 
            gc=0.1, # gap conductance
            g_sparse=0.0,# sparsity of gap junctions
            dt=0.01
    ):
            super().__init__()
            self.W = W
            self.b = b
            self.omega = omega
            self.q_decay = q_decay
            self.b_eff_scale = b_eff_scale
            self.theta = theta
            self.gc = gc
            self.g_sparse = g_sparse
            self.dt = dt

    @staticmethod
    def gap_junction(u, W, gc=0.1, g_sparse=0.0):
        """
        u: (num_cells,) membrane potentials
        W: (num_cells, num_cells) adjacency matrix
        returns gap current for each cell using (u_i - u_j)

        with sparsity
        """
        rowSum = torch.sum(W, dim=1)
        mask = (torch.rand_like(rowSum)> g_sparse).float()
        out = gc * (rowSum * u - torch.matmul(W, u))

        return mask * out

    def BRF_update(self, x, u, v, q):
        '''
        with adaptive threshold: (u_next - theta - q)
        and soft reset: b_eff = b - (q*b_eff_scale)
        '''
        # effective beta for dampening
        b_eff = self.b - (q*self.b_eff_scale)
        u_next = u + (b_eff * u - self.omega * v + x) * self.dt
        v = v + (self.omega * u + b_eff * v) * self.dt

        # spike 
        z = ((u_next - self.theta - q) > 0).float()
        q = self.q_decay * q + z

        return z, u_next, v, q
    
    # computes divergence boundary 
    def sustain_osc(self):
        # arg = torch.clamp(1 - (self.dt * self.omega)**2, min=0.0)
        # return (-1 + torch.sqrt(arg)) / self.dt  
        return (-1 + torch.sqrt(1 - torch.square(self.dt * self.omega))) / self.dt
    
    ### Update function with divergence boubndary
    def BRF_bounded_update(self, x, u, v, q):
        p_omega = self.sustain_osc()

        # DIVERGENCE BOUND & SOFT RESET
        b_eff = p_omega - self.b - q
        # here I add b so the sign stays the same (using b in gui not b_offset to avoid extra inputs)
        # b_eff = p_omega - q + self.b 

        # u_next = u + b_eff * u * self.dt - self.omega * v * self.dt + x * self.dt
        # v = v + self.omega * u * self.dt + b_eff * v * self.dt

        u_next = u + self.dt*(u * b_eff - v * self.omega + x)
        v = v + self.dt*(u * self.omega + v * b_eff)

        z = ((u_next - self.theta - q) > 0)
        # refractory period /adaptive threshold
        q = self.q_decay * q + z

        return z, u_next, v, q

    def run_simulation(
        self,
        dur=5.0,
        I_start=0.0, 
        I_end=0.02, 
        I_amp=10.0
    ):
        num_cells = self.W.shape[0]
        steps = int(dur / self.dt)
        time = torch.arange(steps) * self.dt

        u = torch.zeros(num_cells)
        v = torch.zeros(num_cells)
        q = torch.zeros(num_cells)

        # trajectories
        z_array = torch.zeros(num_cells, steps) 
        u_array = torch.zeros(num_cells, steps)
        v_array = torch.zeros(num_cells, steps)
        q_array = torch.zeros(num_cells, steps)

        # input
        x_array = torch.zeros(steps)
        start_idx = int(I_start / self.dt)
        end_idx = int(I_end / self.dt)
        x_array[start_idx:end_idx] = I_amp

        ### simulation
        for i in range(steps):
            # calculate gap currents
            I_gap = self.gap_junction(u, self.W, self.gc, self.g_sparse)
            # all inputs 
            x_total = x_array[i] + I_gap

            # z, u, v, q = self.BRF_update(x_total, u, v, q)
            z, u, v, q = self.BRF_bounded_update(x_total, u, v, q)

            z_array[:, i] = z
            u_array[:, i] = u
            v_array[:, i] = v
            q_array[:, i] = q

        return time, u_array, v_array, q_array, x_array, z_array



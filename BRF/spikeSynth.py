import numpy as np
import sounddevice as sd      
import soundfile as sf
import torch

class spikeSynth:
    def __init__(
        self,
        z_array: np.ndarray,          # (n_cells, T)
        dt_sim: float,
        Fs: int = 44_100,             # sample rate
        spk_dur: float = 8.0,        # dur of each sine wave for spikes
        f_base: float = 220.0,        # for neuron 0, Hz
        f_step: float = 20.0          # step per neuron index, Hz
    ):
        if isinstance(z_array, torch.Tensor):
            z_array = z_array.detach().cpu().numpy()
        ## take spike array
        self.z = z_array.astype(bool)
        self.dt = dt_sim
        self.Fs = Fs
        self.spk_len = int(spk_dur / 1_000 * Fs)
        self.f_base = f_base
        self.f_step = f_step

        self.audio = self._render()   # precompute waveform

    def _render(self) -> np.ndarray:
        n_cells, T = self.z.shape
        dur_samples = int(T * self.dt * self.Fs)
        out = np.zeros(dur_samples, dtype=np.float32)

        # window Hann smoothing 
        # also try exp decay
        win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.spk_len) / (self.spk_len - 1)))

        # phase increments per cell
        # maybe another time ill rand assign of something else idk
        freqs = self.f_base + self.f_step * np.arange(n_cells)
        ph_incr = 2 * np.pi * freqs / self.Fs

        for i in range(n_cells):
            spike_indices = np.where(self.z[i])[0] # indices in simulation time
            sample_idx = (spike_indices * self.dt * self.Fs).astype(int) # maps to sample time

            for s in sample_idx:
                # ensure stay in bounds, add spk len
                end = min(s + self.spk_len, dur_samples)
                length = end - s
                t = np.arange(length)

                out[s:end] += np.sin(ph_incr[i] * t)[:length] * win[:length]

        # prevent clipping
        out /= np.max(np.abs(out)) + 1e-12
        return out

    def save_wav(self, path: str):
        sf.write(path, self.audio, self.Fs, subtype='PCM_16')


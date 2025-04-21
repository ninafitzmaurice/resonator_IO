TODO: 
add IO cells to microcircuit

BRF_io: 
- divergence boundary:
---- chose way to formulate this to match IO cells: what range is going to be useful in learning context 
- surrogate gradient to BRF cell: jax 

LATER, NOT IMPORTANT
- spike train inputs - set % of random neurons to get input
- test different spike freq inputs? need a smart way to do systematically
- add distant dependent decay? 

GUI:
toggle on/off plotting adaptive threshold 
sound: toggle spikes or oscillations or off


TO THINK:
how am I gonna implement into the wider microcircuit
1. without surrogate gradient
2. with surrogate gradient 


# BAlANCED RESONATE AND FIRE IO CELLS COUPLED VIA GAP JUNCTIONS 

divergence boundary:
p_omega = (-1 + np.sqrt(1 - (dt*omega)**2)) / dt

in the code...
def sustain_osc(omega: torch.Tensor, dt: float = DEFAULT_DT) -> torch.Tensor:
    return (-1 + torch.sqrt(1 - torch.square(dt * omega))) / dt
p_omega = sustain_osc(omega)

soft reset with divergence boundary:
b = p_omega - b_offset - q
this includes q, refractory variable with decay and b_offset which i wrote as a scale for q instead...
all this does is bring the system down below the divergence boundary so b does not explode


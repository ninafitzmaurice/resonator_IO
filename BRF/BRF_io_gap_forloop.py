import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QSlider, QLineEdit
from PyQt5.QtCore import Qt
import pyqtgraph as pg

DUR=10.0
I_START=0.0 
I_END=5.0
I_AMP=1.0

THETA=0.5

def gap_junc(Vd, Vde, gc=0.1, sigma=50, base_conductance=0.4):
    V_diff = Vd - Vde
    fV = 0.6 * torch.exp(V_diff**2 / sigma**2) + base_conductance
    # Ic = gc * fV * V_diff
    Ic = gc * V_diff
    return Ic

def BRF_update(
    x,      # input current
    u,      # real part mem potential
    v,      # imaginary part mem pot
    q,      # refractory term
    b,      # damping factor / attraction to resting state
    omega,  # angular frequency, oscillations
    r_decay=0.9,
    theta=THETA,
    dt=0.01
):
    ## effective beta is what remains after refractory period
    b_eff = b - q

    u_next = u + b_eff*u*dt - omega*v*dt + x*dt
    v_next = v + omega*u*dt + b_eff*v*dt

    z = ((u_next - theta - q) > 0).float()

    # q_next = r_decay*q + z
    q_next = r_decay*q

    return z, u_next, v_next, q_next


def BRF_simulate(
    dur=DUR, I_start=I_START, I_end=I_END, I_amp=I_AMP, 
    b1=0.1, omega1=6.0, b2=0.1, omega2=6.0,
    r_decay=0.9, theta=1.0, dt=0.01, gc=0.1
):
    steps = int(dur / dt)
    time = torch.arange(steps) * dt

    u1_array, v1_array, q1_array = torch.zeros(steps), torch.zeros(steps), torch.zeros(steps)
    u2_array, v2_array, q2_array = torch.zeros(steps), torch.zeros(steps), torch.zeros(steps)
    x_array = torch.zeros(steps)

    u1, v1, q1 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    u2, v2, q2 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    start_idx = int(I_start/dt)
    end_idx = int(I_end/dt)
    x_array[start_idx:end_idx] = I_amp

    for i in range(steps):
        ## calculate gap input
        gap_current = gap_junc(u2, u1, gc)
        ### input + gap inputs in update 
        z1, u1, v1, q1 = BRF_update(x_array[i]+gap_current, u1, v1, q1, b1, omega1, r_decay, theta, dt)
        z2, u2, v2, q2 = BRF_update(x_array[i]-gap_current, u2, v2, q2, b2, omega2, r_decay, theta, dt)

        # u1_array[i], v1_array[i], q1_array[i] = u1, v1, q1
        # u2_array[i], v2_array[i], q2_array[i] = u2, v2, q2

        u1_array[i], v1_array[i], q1_array[i] = u1 + q1, v1, q1
        u2_array[i], v2_array[i], q2_array[i] = u2 + q2, v2, q2
    
    return time, u1_array, v1_array, q1_array, u2_array, v2_array, q2_array, x_array


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRF Neuron Simulation with Gap Junction")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        self.graphWidget = pg.PlotWidget()
        layout.addWidget(self.graphWidget)
        self.graphWidget.setLabel("left", "Membrane Potential (u)")
        self.graphWidget.setLabel("bottom", "Time (s)")
        
        self.b1_slider = self.create_slider("b1", layout, 0, 100, 90, lambda x: -10 + (x / 100) * 11)
        self.b2_slider = self.create_slider("b2", layout, 0, 100, 90, lambda x: -10 + (x / 100) * 11)
        self.omega1_slider = self.create_slider("omega1", layout, 1, 20, 10)
        self.omega2_slider = self.create_slider("omega2", layout, 1, 20, 10)
        self.gc_slider = self.create_slider("Gap Conductance", layout, 0, 100, 10, lambda x: x / 100.0)
        
        # self.update_button = QPushButton("Update Plot")
        # self.update_button.clicked.connect(self.update_plot)
        # layout.addWidget(self.update_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.update_plot()

    # def create_slider(self, name, layout, min_val, max_val, default, transform_fn=lambda x: x):
    #     value = transform_fn(default)
    #     label = QLabel(f"{name}: {value:.2f}")
    #     slider = QSlider(Qt.Horizontal)
    #     slider.setMinimum(min_val)
    #     slider.setMaximum(max_val)
    #     slider.setValue(default)
    #     slider.valueChanged.connect(lambda: label.setText(f"{name}: {transform_fn(slider.value()):.2f}"))
    #     layout.addWidget(label)
    #     layout.addWidget(slider)
    #     return slider

    def create_slider(self, name, layout, min_val, max_val, default, transform_fn=lambda x: x):
        value = transform_fn(default)
        label = QLabel(f"{name}: {value:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)

        def on_slider_change():
            val = transform_fn(slider.value())
            label.setText(f"{name}: {val:.2f}")
            self.update_plot()

        slider.valueChanged.connect(on_slider_change)
        layout.addWidget(label)
        layout.addWidget(slider)
        return slider


    def update_plot(self):
        b1 = -10 + (self.b1_slider.value() / 100.0) * 11
        b2 = -10 + (self.b2_slider.value() / 100.0) * 11
        omega1 = self.omega1_slider.value()
        omega2 = self.omega2_slider.value()
        gc = self.gc_slider.value() / 100.0

        time, u1, v1, q1, u2, v2, q2, x_array = BRF_simulate(b1=b1, omega1=omega1, b2=b2, omega2=omega2, gc=gc)
        
        self.graphWidget.clear()
        self.graphWidget.plot(time.numpy(), u1.numpy(), pen='cyan', name="Neuron 1")
        self.graphWidget.plot(time.numpy(), u2.numpy(), pen='pink', name="Neuron 2")
        x_scaled = x_array / x_array.max() * max(u1.max(), u2.max()).item() if x_array.max() > 0 else x_array
        self.graphWidget.plot(time.numpy(), x_scaled.numpy(), pen=pg.mkPen('g', style=Qt.DashLine), name="Input Current")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

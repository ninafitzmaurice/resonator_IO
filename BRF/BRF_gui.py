import sys
import json
import numpy as np

import torch
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSlider, QLabel, QTextEdit, QLineEdit
)

#TODO: add spike to mem potential and soft reset from there? visualses spiking dynamics better (use double guass?)
#TODO: add input spike trains instead of current (can test response to diff frequencies)
#TODO: THINK: how to implement gap junctions to BRF cells? 
#TODO: THINK: does the (smooth) reset influence dynamics in the context of IO cells? what reset do I need in IO cells?

def BRF_update(
    x,      # input current
    u,      # real part mem potential
    v,      # imaginary part mem pot
    q,      # refractory term
    b,      # damping factor / attraction to resting state
    omega,  # angular frequency, oscillations
    r_decay=0.9,
    theta=1.0,
    dt=0.01
):
    # effective beta val for smooth reset based on refractory period
    # implementation for IO?
    b_eff = b - q
    # 1) update real part 
    u_next = u + b_eff*u*dt - omega*v*dt + x*dt
    # 2) update complex part
    v_next = v + omega*u*dt + b_eff*v*dt
    # spike
    z = ((u_next - theta - q) > 0).float()
    # refractry period
    q_next = r_decay*q + z
    return z, u_next, v_next, q_next

def BRF_simulate(
    dur=2.0,
    I_start=0.0,
    I_end=0.02,
    I_amp=1.0,
    b_val=0.1,
    omega_val=6.0,
    r_decay_val=0.9,
    theta_val=1.0,
    dt=0.01
):
    steps = int(dur / dt)
    time = torch.arange(steps) * dt

    x_array = torch.zeros(steps)
    u_array = torch.zeros(steps)
    v_array = torch.zeros(steps)
    q_array = torch.zeros(steps)
    z_array = torch.zeros(steps)

    u = torch.tensor(0.0)
    v = torch.tensor(0.0)
    q = torch.tensor(0.0)

    start_idx = int(I_start/dt)
    end_idx = int(I_end/dt)
    x_array[start_idx:end_idx] = I_amp

    for i in range(steps):
        z, u, v, q = BRF_update(x_array[i], u, v, q,
                                b_val, omega_val,
                                r_decay_val, theta_val,
                                dt=dt)
        u_array[i] = u
        v_array[i] = v
        q_array[i] = q
        z_array[i] = z

    return time, x_array, u_array, v_array, q_array, z_array

###############################################################################
# GUI
###############################################################################
params_default = dict(
    b_val       = 0.1,
    omega_val   = 6.0,
    r_decay_val = 0.9,
    theta_val   = 0.5
)

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('''
            background-color: #000000;
            color: #ffffff;
        ''')
        self.init_ui()
        self.update_plot()

    def init_ui(self):
        self.setWindowTitle('BRF Neuron')
        self.setGeometry(100, 100, 800, 800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # plot, title
        self.toplabel = QLabel('BRF Simulation')
        self.layout.addWidget(self.toplabel)
        self.graphWidget = PlotWidget()
        self.layout.addWidget(self.graphWidget)

        # Text
        form_layout = QFormLayout()
        self.layout.addLayout(form_layout)

        self.dur_edit    = QLineEdit("2.0")
        self.Istart_edit = QLineEdit("0.5")
        self.Iend_edit   = QLineEdit("1.0")
        self.Iamp_edit   = QLineEdit("1.0")

        form_layout.addRow(QLabel("Duration"),  self.dur_edit)
        form_layout.addRow(QLabel("I_start"),   self.Istart_edit)
        form_layout.addRow(QLabel("I_end"),     self.Iend_edit)
        form_layout.addRow(QLabel("I_amp"),     self.Iamp_edit)

        self.dur_edit.textChanged.connect(self.update_plot)
        self.Istart_edit.textChanged.connect(self.update_plot)
        self.Iend_edit.textChanged.connect(self.update_plot)
        self.Iamp_edit.textChanged.connect(self.update_plot)

        # Sliders
        sliders_layout = QHBoxLayout()
        sliders_left   = QFormLayout()
        sliders_right  = QFormLayout()
        self.layout.addLayout(sliders_layout)
        sliders_layout.addLayout(sliders_left)
        sliders_layout.addLayout(sliders_right)

        self.sliders = {}
        self.slider_labels = {}

        slider_params = [
            ("b_val",       "Damping (b)",        0.1),
            ("omega_val",   "Omega (ω)",          6.0),
            ("r_decay_val", "Refractory Decay",   0.9),
            ("theta_val",   "Threshold (θ)",      0.5)
        ]
        param_max_values = {
            "b_val":       500,
            "omega_val":   500,
            "r_decay_val": 500,
            "theta_val":   500
        }

        for i, (key, label_text, default_v) in enumerate(slider_params):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(param_max_values[key])
            slider.setValue(int(param_max_values[key] * 0.2))  # naive init

            label = QLabel(f"{label_text} ({default_v})")
            self.sliders[key] = slider
            self.slider_labels[key] = label

            # distribute
            if i % 2 == 0:
                sliders_left.addRow(label, slider)
            else:
                sliders_right.addRow(label, slider)

            slider.valueChanged.connect(self.update_plot)

        # Buttons
        btn_layout = QHBoxLayout()
        reset_button = QPushButton('Reset')
        reset_button.clicked.connect(self.on_reset)
        btn_layout.addWidget(reset_button)

        randomize_button = QPushButton('Randomize')
        randomize_button.clicked.connect(self.on_randomize)
        btn_layout.addWidget(randomize_button)

        sliders_left.addRow(btn_layout)

        # Text area for param display
        self.textedit_params = QTextEdit()
        self.textedit_params.setReadOnly(True)
        self.layout.addWidget(self.textedit_params)

        self.show()

    def on_reset(self):
        self.dur_edit.setText("2.0")
        self.Istart_edit.setText("0.5")
        self.Iend_edit.setText("1.0")
        self.Iamp_edit.setText("1.0")

        for k, v in params_default.items():
            slider = self.sliders[k]
            slider.setValue(int(slider.maximum()*0.2))
        self.update_plot()

    def on_randomize(self):
        for key in self.sliders:
            slider = self.sliders[key]
            val = np.random.randint(0, slider.maximum()+1)
            slider.setValue(val)
        self.update_plot()

    def read_text_params(self):
        try:
            dur    = float(self.dur_edit.text())
            Istart = float(self.Istart_edit.text())
            Iend   = float(self.Iend_edit.text())
            Iamp   = float(self.Iamp_edit.text())
        except ValueError:
            dur, Istart, Iend, Iamp = 2.0, 0.5, 1.0, 1.0
        return dur, Istart, Iend, Iamp

    def read_slider_params(self):
        """
        Map each slider's integer [0..500] to a range that includes 0 exactly.

        b_val:    from -10..+10 => total span 20
                  fraction = i/500
                  b_val = -10 + (20 * fraction)
                  => If i=250 => fraction=0.5 => b=0 exactly
        omega_val from 0..100 => total span 100
                  fraction = i/500
                  => i=0 => fraction=0 => 0, i=500 => fraction=1 => 100
        r_decay in  [0..1]
                  fraction = i/500 => direct => 0..1
        theta in [0..1]
                  fraction = i/500 => direct => 0..1
        """
        i_b = self.sliders["b_val"].value()
        frac_b = i_b / 500.0
        b_val = -10.0 + 20.0 * frac_b  # => -10..+10

        i_omega = self.sliders["omega_val"].value()
        frac_omega = i_omega / 500.0
        omega_val = 100.0 * frac_omega  # => 0..100

        i_rdec = self.sliders["r_decay_val"].value()
        frac_rdec = i_rdec / 500.0
        r_decay_val = frac_rdec  # => 0..1

        i_theta = self.sliders["theta_val"].value()
        frac_theta = i_theta / 500.0
        theta_val = frac_theta  # => 0..1

        return b_val, omega_val, r_decay_val, theta_val

    def update_plot(self):
        dur, Istart, Iend, Iamp = self.read_text_params()
        b_val, omega_val, r_decay_val, theta_val = self.read_slider_params()

        # Update slider labels
        self.slider_labels["b_val"].setText(f"Damping (b) ({b_val:.3f})")
        self.slider_labels["omega_val"].setText(f"Omega (ω) ({omega_val:.3f})")
        self.slider_labels["r_decay_val"].setText(f"Refractory Decay ({r_decay_val:.3f})")
        self.slider_labels["theta_val"].setText(f"Threshold (θ) ({theta_val:.3f})")

        try:
            time, x_array, u_array, v_array, q_array, z_array = BRF_simulate(
                dur=dur,
                I_start=Istart,
                I_end=Iend,
                I_amp=Iamp,
                b_val=b_val,
                omega_val=omega_val,
                r_decay_val=r_decay_val,
                theta_val=theta_val,
                dt=0.01
            )
        except Exception as e:
            self.toplabel.setText(f"Error: {repr(e)}")
            self.graphWidget.clear()
            return

        time    = time.cpu().numpy()
        x_array = x_array.cpu().numpy()
        u_array = u_array.cpu().numpy()
        v_array = v_array.cpu().numpy()
        # print(v_array)
        q_array = q_array.cpu().numpy()

        export_dict = dict(
            duration=dur,
            I_start=Istart,
            I_end=Iend,
            I_amp=Iamp,
            b_val=b_val,
            omega_val=omega_val,
            r_decay_val=r_decay_val,
            theta_val=theta_val,
            dt=0.01,
            ### ?? NOT SURE IF THIS IS RIGHT
            freq_Hz=(omega_val / (2 * np.pi))
        )
        s = json.dumps(export_dict, indent=2)
        self.textedit_params.setText(s)

        self.graphWidget.clear()
        self.graphWidget.addLegend()

        # Real part
        self.graphWidget.plot(
            time, u_array,
            pen=pg.mkPen(color=(255, 192, 203), width=2),
            name="Real Potential"
        )
        # Imag part
        self.graphWidget.plot(
            time, v_array,
            pen=pg.mkPen(color=(128, 0, 128), width=2),
            name="Imag Potential"
        )
        # Input current
        self.graphWidget.plot(
            time, x_array,
            pen=pg.mkPen(color=(255, 165, 0), width=2),
            name="Input Current"
        )
        # Refractory
        self.graphWidget.plot(
            time, q_array,
            pen=pg.mkPen(color=(173, 216, 230), style=Qt.DotLine, width=2),
            name="Refractory"
        )

        self.graphWidget.setLabels(left='Amplitude', bottom='Time')
        self.toplabel.setText('BRF simulation updated')

def main():
    app = QApplication([])
    window = Window()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

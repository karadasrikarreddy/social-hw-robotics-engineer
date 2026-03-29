"""
simulate_haptic_pid.py
======================
Python simulation of haptic_pid.c — mirrors the C logic exactly.

Produces 4 plots:
  1. Step response  (position + torque over time)
  2. Anti-windup    (integral growth with/without anti-windup)
  3. Detent force profile  (torque vs knob angle for one full detent span)
  4. Derivative filter     (raw vs filtered D-term under a noise spike)

Run:
    pip install matplotlib numpy
    python simulate_haptic_pid.py

Author: Srikar Reddy
Task  : Social HW Lab — Robotics Engineer Assessment, Task 3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Constants (match haptic_pid.c) ──────────────────────────────────────────
DT           = 1e-4        # 10 kHz control loop
MAX_OUT      =  1.0
MIN_OUT      = -1.0
LPF_ALPHA    = 0.1118      # cutoff 200 Hz @ 10 kHz

KP, KI, KD  = 8.0, 0.5, 0.02
DETENT_N     = 24
DETENT_SPACE = 2 * np.pi / DETENT_N


# ─── Python PID (mirrors C struct + functions) ────────────────────────────────
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral   = 0.0
        self.prev_error = 0.0
        self.d_filtered = 0.0
        self.saturated  = False
        self.sat_sign   = 0.0

    def reset(self):
        self.integral = self.prev_error = self.d_filtered = 0.0
        self.saturated = False; self.sat_sign = 0.0

    def update(self, setpoint, measured):
        error  = setpoint - measured
        p_term = self.Kp * error

        # Anti-windup: conditional integration
        integrate = (not self.saturated) or \
                    (self.saturated and error * self.sat_sign < 0)
        if integrate:
            self.integral += error * DT
        i_term = self.Ki * self.integral

        # Filtered derivative
        d_raw           = (error - self.prev_error) / DT
        self.d_filtered = LPF_ALPHA * d_raw + (1 - LPF_ALPHA) * self.d_filtered
        d_term          = self.Kd * self.d_filtered

        output_raw = p_term + i_term + d_term
        output     = np.clip(output_raw, MIN_OUT, MAX_OUT)

        self.saturated = (output >= MAX_OUT or output <= MIN_OUT)
        self.sat_sign  = np.sign(output) if self.saturated else 0.0
        self.prev_error = error
        return output


# ─── Simple plant: velocity += torque/inertia·dt, pos += vel·dt ──────────────
class Plant:
    def __init__(self, pos0=0.0, inertia=0.005, friction=0.002):
        self.pos = pos0; self.vel = 0.0
        self.inertia = inertia; self.friction = friction

    def step(self, torque):
        accel    = (torque - self.friction * self.vel) / self.inertia
        self.vel += accel * DT
        self.pos += self.vel * DT


def detent_setpoint(angle, spacing=DETENT_SPACE):
    return round(angle / spacing) * spacing

def endstop_setpoint(desired, mn, mx):
    return np.clip(desired, mn, mx)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Step response
# ─────────────────────────────────────────────────────────────────────────────
def sim_step_response(n_steps=20000):   # 200 ms at 10 kHz — shows full settling
    pid = PIDController(KP, KI, KD)
    plant = Plant()
    target = DETENT_SPACE   # one detent away

    t, pos, torque = [], [], []
    for i in range(n_steps):
        u = pid.update(target, plant.pos)
        plant.step(u)
        t.append(i * DT * 1000)   # ms
        pos.append(plant.pos)
        torque.append(u)
    return np.array(t), np.array(pos), np.array(torque), target


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Anti-windup comparison
# ─────────────────────────────────────────────────────────────────────────────
def sim_anti_windup(n_stuck=800, n_free=400):
    """
    Phase 1 (0…n_stuck): plant frozen at 0, setpoint=0.3.
                         Integrators wind up.
    Phase 2 (n_stuck…end): plant released, runs freely.
    Compare integral and position trajectory with vs without anti-windup.
    """
    setpoint = 0.3

    results = {}
    for label, use_aw in [("With Anti-Windup", True), ("Without Anti-Windup", False)]:
        pid = PIDController(KP, KI, KD)
        plant = Plant()
        integrals, positions, torques = [], [], []

        for i in range(n_stuck + n_free):
            if i < n_stuck:
                meas = 0.0   # plant frozen
                u = pid.update(setpoint, meas)
                if not use_aw:
                    # Force integration regardless of saturation
                    pid.integral += (setpoint - meas) * DT
                    pid.saturated = False
            else:
                u = pid.update(setpoint, plant.pos)
                plant.step(u)

            integrals.append(pid.integral)
            positions.append(plant.pos if i >= n_stuck else 0.0)
            torques.append(u)

        t = np.arange(n_stuck + n_free) * DT * 1000
        results[label] = (t, np.array(integrals), np.array(positions), np.array(torques))
    return results, n_stuck * DT * 1000, setpoint


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Detent force profile
# ─────────────────────────────────────────────────────────────────────────────
def sim_detent_force():
    """
    Sweep knob angle slowly across 2 detent spans.
    At each position compute the torque the PID would command.
    Shows the spring-like restoring force between detents.
    """
    angles  = np.linspace(0, 2 * DETENT_SPACE, 500)
    torques = []
    for a in angles:
        sp  = detent_setpoint(a)
        err = sp - a
        # Proportional-only for steady-state profile (no dynamics)
        torques.append(np.clip(KP * err, MIN_OUT, MAX_OUT))
    return angles, np.array(torques)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Derivative filter noise rejection
# ─────────────────────────────────────────────────────────────────────────────
def sim_derivative_filter(n_steps=200, spike_step=100, spike_mag=0.05):
    setpoint = 0.2618
    position = 0.26

    pid_raw  = PIDController(KP, 0, KD)
    pid_filt = PIDController(KP, 0, KD)

    raw_d, filt_d = [], []
    for i in range(n_steps):
        noise = spike_mag if i == spike_step else 0.0
        meas  = position + noise

        # Raw: compute d_raw manually without filtering
        err_r = setpoint - meas
        d_r   = (err_r - pid_raw.prev_error) / DT
        raw_d.append(d_r)
        pid_raw.prev_error = err_r

        # Filtered
        pid_filt.update(setpoint, meas)
        filt_d.append(pid_filt.d_filtered)

    t = np.arange(n_steps) * DT * 1000
    return t, np.array(raw_d), np.array(filt_d), spike_step * DT * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Main — build all 4 plots
# ─────────────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Haptic Dial PID — Simulation Results\n"
                 "Social HW Lab — Robotics Assessment Task 3",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── Plot 1: Step Response ─────────────────────────────────────────────────
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = ax1a.twinx()

    t, pos, trq, target = sim_step_response()
    ax1a.plot(t, np.degrees(pos),    color="#185FA5", lw=2,   label="Position (°)")
    ax1a.axhline(np.degrees(target), color="#185FA5", lw=1,
                 ls="--", alpha=0.5, label=f"Target = {np.degrees(target):.1f}°")
    ax1b.plot(t, trq, color="#E24B4A", lw=1.2, alpha=0.7, label="Torque")
    ax1a.set_xlabel("Time (ms)"); ax1a.set_ylabel("Position (°)", color="#185FA5")
    ax1b.set_ylabel("Torque (norm.)", color="#E24B4A")
    ax1a.set_title("1 — Step Response (one detent)")
    lines1 = ax1a.get_lines() + ax1b.get_lines()
    ax1a.legend(lines1, [l.get_label() for l in lines1], fontsize=8, loc="center right")
    ax1a.set_xlim(0, 200); ax1a.grid(True, alpha=0.3)

    # ── Plot 2: Anti-Windup ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    results, release_t, sp = sim_anti_windup()
    colors = {"With Anti-Windup": "#3B6D11", "Without Anti-Windup": "#E24B4A"}
    for label, (t, intg, pos2, trq2) in results.items():
        ax2.plot(t, intg, lw=2, color=colors[label], label=label)
    ax2.axvline(release_t, color="gray", ls=":", lw=1.5, label="Plant released")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Integral accumulation")
    ax2.set_title("2 — Anti-Windup: Integral at Hard-Stop")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Plot 3: Detent Force Profile ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    angles, forces = sim_detent_force()
    ax3.plot(np.degrees(angles), forces, color="#185FA5", lw=2)
    ax3.fill_between(np.degrees(angles), forces, alpha=0.15, color="#185FA5")
    for k in range(3):
        ax3.axvline(np.degrees(k * DETENT_SPACE), color="gray", ls="--",
                    lw=1, label="Detent centre" if k == 0 else "")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_xlabel("Knob Angle (°)"); ax3.set_ylabel("Restoring Torque (norm.)")
    ax3.set_title("3 — Detent Force Profile (Kp spring)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── Plot 4: Derivative Filter ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    t4, raw_d, filt_d, spike_t = sim_derivative_filter()

    # Clip raw for visibility (spike is huge)
    raw_clipped = np.clip(raw_d, -2000, 2000)
    ax4.plot(t4, raw_clipped,  color="#E24B4A", lw=1.2, alpha=0.8, label="Raw D-term (clipped ±2000)")
    ax4.plot(t4, filt_d,       color="#3B6D11", lw=2,               label=f"Filtered D (f_c=200 Hz)")
    ax4.axvline(spike_t, color="gray", ls=":", lw=1.5, label="Noise spike")
    ax4.set_xlabel("Time (ms)"); ax4.set_ylabel("Derivative (rad/s)")
    ax4.set_title("4 — D-Term: Raw vs Low-Pass Filtered")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    plt.savefig("haptic_pid_simulation.png", dpi=150, bbox_inches="tight")
    print("[Sim] Saved → haptic_pid_simulation.png")
    plt.show()


if __name__ == "__main__":
    main()

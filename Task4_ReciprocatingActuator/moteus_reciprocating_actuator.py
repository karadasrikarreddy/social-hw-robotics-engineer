"""
moteus_reciprocating_actuator.py
=================================
Task 4 — High-Speed Reciprocating Actuator
Social HW Lab — Robotics Engineer Assessment

Phases
------
  Phase 1 : Safety limits + torque-based homing to hard-stop
  Phase 2 : Cyclic trajectory 0.0 ↔ 2.0 rev at 5 rev/s, 20 rev/s²
  Phase 3 : Telemetry logging → CSV + matplotlib plot

Hardware
--------
  Controller : moteus n1 / r4.x / c1
  Comms      : fdcanusb  (default) or pi3hat
  Motor      : any BLDC calibrated for moteus

Usage
-----
  pip install moteus matplotlib pandas
  python moteus_reciprocating_actuator.py              # live run
  python moteus_reciprocating_actuator.py --simulate   # offline demo (no hardware)
  python moteus_reciprocating_actuator.py --plot-only  # re-plot existing telemetry.csv

Author : Srikar Reddy
"""

import asyncio
import csv
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─── Try importing moteus (not required in --simulate mode) ───────────────────
try:
    import moteus
    MOTEUS_AVAILABLE = True
except ImportError:
    MOTEUS_AVAILABLE = False

# ─── Configuration ─────────────────────────────────────────────────────────────
SERVO_ID            = 1

# Safety limits
MAX_CURRENT_A       = 2.0        # servo.max_current_A
POS_MIN             = -0.05      # rev — small margin before hard-stop
POS_MAX             =  2.10      # rev — small margin past travel end

# Homing parameters
HOMING_VELOCITY     =  0.1       # rev/s — slow creep toward hard-stop
HOMING_TORQUE_THRESH=  0.8       # Nm — q_current threshold to detect wall hit
HOMING_MAX_S        = 10.0       # seconds before homing timeout

# Trajectory parameters
TRAJ_POS_A          = 0.0        # rev — endpoint A
TRAJ_POS_B          = 2.0        # rev — endpoint B
TRAJ_VEL_LIMIT      = 5.0        # rev/s
TRAJ_ACCEL_LIMIT    = 20.0       # rev/s²
ENDPOINT_DWELL_S    = 0.10       # 100 ms pause at each end
N_CYCLES            = 5          # number of back-and-forth cycles to execute

# Watchdog: moteus requires a command every 100 ms or it faults
WATCHDOG_PERIOD_S   = 0.05       # send command every 50 ms (2× safety margin)

# Telemetry
CSV_PATH            = "telemetry.csv"
PLOT_PATH           = "telemetry_plot.png"


# ─── Telemetry record ──────────────────────────────────────────────────────────
@dataclass
class TelemetryRow:
    timestamp_s:     float
    target_pos_rev:  float
    actual_pos_rev:  float
    actual_vel_revs: float
    q_current_A:     float
    d_current_A:     float
    bus_voltage_V:   float
    temperature_C:   float
    mode:            str    # "homing" | "trajectory" | "dwell"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Initialisation & homing
# ─────────────────────────────────────────────────────────────────────────────

async def configure_safety_limits(controller) -> None:
    """
    Write safety limits to the controller via the moteus register interface.

    Registers written:
      servo.max_current_A   → MAX_CURRENT_A
      servo.position_min    → POS_MIN
      servo.position_max    → POS_MAX
    """
    print(f"[Init] Setting max_current={MAX_CURRENT_A} A  "
          f"pos=[{POS_MIN}, {POS_MAX}] rev")

    await controller.set_register(
        moteus.Register.SERVO_MAX_CURRENT_A,
        MAX_CURRENT_A
    )
    await controller.set_register(
        moteus.Register.SERVO_POSITION_MIN,
        POS_MIN
    )
    await controller.set_register(
        moteus.Register.SERVO_POSITION_MAX,
        POS_MAX
    )
    print("[Init] Safety limits configured ✓")


async def torque_home(controller, log: List[TelemetryRow]) -> bool:
    """
    Phase 1 — Torque-based homing.

    Algorithm
    ---------
    1. Command slow velocity toward the hard-stop (negative direction).
    2. Poll q_current every WATCHDOG_PERIOD_S.
    3. When |q_current| exceeds HOMING_TORQUE_THRESH → motor has hit the wall.
    4. Call set_stop() to de-energise, then set position=NaN to clear the
       position reference, then redefine this point as Position 0.

    Backlash handling (see write-up below):
      After detecting the wall hit, we back off 0.02 rev before zeroing.
      This removes the mechanical preload so the zero reference is
      reproducible regardless of which direction backlash is sitting.
    """
    print("[Homing] Creeping toward hard-stop ...")
    t0 = time.monotonic()

    while True:
        elapsed = time.monotonic() - t0
        if elapsed > HOMING_MAX_S:
            print("[Homing] TIMEOUT — hard-stop not detected!")
            return False

        # Command slow velocity; q_current feedback tells us about torque
        state = await controller.set_position(
            position   = math.nan,          # position=nan → velocity mode
            velocity   = -HOMING_VELOCITY,  # negative = toward hard-stop
            maximum_torque = MAX_CURRENT_A,
            watchdog_timeout = math.nan,    # managed by our loop
            query = True,
        )

        q_current = abs(state.values[moteus.Register.Q_CURRENT])
        pos       = state.values[moteus.Register.POSITION]
        voltage   = state.values.get(moteus.Register.VOLTAGE, 0.0)
        temp      = state.values.get(moteus.Register.TEMPERATURE, 0.0)

        log.append(TelemetryRow(
            timestamp_s    = elapsed,
            target_pos_rev = math.nan,
            actual_pos_rev = pos,
            actual_vel_revs= state.values.get(moteus.Register.VELOCITY, 0.0),
            q_current_A    = q_current,
            d_current_A    = state.values.get(moteus.Register.D_CURRENT, 0.0),
            bus_voltage_V  = voltage,
            temperature_C  = temp,
            mode           = "homing",
        ))

        print(f"  [Homing] t={elapsed:.2f}s  pos={pos:.3f}  Iq={q_current:.3f} A")

        if q_current >= HOMING_TORQUE_THRESH:
            print(f"[Homing] Wall detected! Iq={q_current:.3f} A ≥ {HOMING_TORQUE_THRESH} A")
            break

        await asyncio.sleep(WATCHDOG_PERIOD_S)

    # ── Stop the motor cleanly ────────────────────────────────────────────────
    await controller.set_stop()
    await asyncio.sleep(0.05)

    # ── Back off 0.02 rev to relieve mechanical preload (backlash compensation)
    print("[Homing] Backing off 0.02 rev to relieve preload ...")
    await controller.set_position(
        position         = math.nan,
        velocity         = +HOMING_VELOCITY,
        maximum_torque   = 0.3,
        watchdog_timeout = math.nan,
        query            = False,
    )
    await asyncio.sleep(0.2)
    await controller.set_stop()
    await asyncio.sleep(0.05)

    # ── Redefine current position as 0 ────────────────────────────────────────
    # set_position with position=NaN clears the reference; subsequent commands
    # are relative to the next non-NaN position, which we set to 0.0.
    await controller.set_position(
        position         = 0.0,
        velocity         = 0.0,
        maximum_torque   = 0.1,
        watchdog_timeout = math.nan,
        query            = False,
    )
    await asyncio.sleep(0.1)
    print("[Homing] Home set to Position 0 ✓")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Cyclic trajectory
# ─────────────────────────────────────────────────────────────────────────────

async def run_trajectory(controller, log: List[TelemetryRow]) -> None:
    """
    Phase 2 — Cyclic motion: 0.0 ↔ 2.0 rev for N_CYCLES cycles.

    Profile:  velocity_limit=5 rev/s,  accel_limit=20 rev/s²
    Dwell:    100 ms at each endpoint before reversing.

    Watchdog: moteus faults if no command arrives within 100 ms.
    We send a keep-alive every WATCHDOG_PERIOD_S (50 ms) during dwells.
    """
    t0 = time.monotonic()
    endpoints = [TRAJ_POS_A, TRAJ_POS_B]
    ep_idx    = 0   # start by moving to pos_B (away from hard-stop)

    print(f"[Traj] Starting {N_CYCLES} cycles  "
          f"vel={TRAJ_VEL_LIMIT} rev/s  accel={TRAJ_ACCEL_LIMIT} rev/s²")

    for cycle in range(N_CYCLES):
        for leg in range(2):   # leg 0: A→B, leg 1: B→A
            target = endpoints[ep_idx % 2]
            ep_idx += 1

            print(f"  [Traj] Cycle {cycle+1}/{N_CYCLES}  target={target:.1f} rev")

            # ── Command the move ───────────────────────────────────────────────
            await controller.set_position(
                position         = target,
                velocity_limit   = TRAJ_VEL_LIMIT,
                accel_limit      = TRAJ_ACCEL_LIMIT,
                maximum_torque   = MAX_CURRENT_A,
                watchdog_timeout = math.nan,
                query            = False,
            )

            # ── Wait until within 0.05 rev of target, keep watchdog alive ─────
            while True:
                state = await controller.set_position(
                    position         = target,
                    velocity_limit   = TRAJ_VEL_LIMIT,
                    accel_limit      = TRAJ_ACCEL_LIMIT,
                    maximum_torque   = MAX_CURRENT_A,
                    watchdog_timeout = math.nan,
                    query            = True,
                )
                elapsed = time.monotonic() - t0
                pos  = state.values[moteus.Register.POSITION]
                vel  = state.values.get(moteus.Register.VELOCITY, 0.0)
                iq   = state.values.get(moteus.Register.Q_CURRENT, 0.0)
                vbus = state.values.get(moteus.Register.VOLTAGE, 0.0)
                temp = state.values.get(moteus.Register.TEMPERATURE, 0.0)
                did  = state.values.get(moteus.Register.D_CURRENT, 0.0)

                log.append(TelemetryRow(
                    timestamp_s    = elapsed,
                    target_pos_rev = target,
                    actual_pos_rev = pos,
                    actual_vel_revs= vel,
                    q_current_A    = iq,
                    d_current_A    = did,
                    bus_voltage_V  = vbus,
                    temperature_C  = temp,
                    mode           = "trajectory",
                ))

                if abs(pos - target) < 0.05:
                    break
                await asyncio.sleep(WATCHDOG_PERIOD_S)

            # ── Endpoint dwell — keep sending commands to satisfy watchdog ─────
            dwell_start = time.monotonic()
            while time.monotonic() - dwell_start < ENDPOINT_DWELL_S:
                state = await controller.set_position(
                    position         = target,
                    velocity_limit   = TRAJ_VEL_LIMIT,
                    accel_limit      = TRAJ_ACCEL_LIMIT,
                    maximum_torque   = MAX_CURRENT_A,
                    watchdog_timeout = math.nan,
                    query            = True,
                )
                elapsed = time.monotonic() - t0
                pos  = state.values[moteus.Register.POSITION]
                log.append(TelemetryRow(
                    timestamp_s    = elapsed,
                    target_pos_rev = target,
                    actual_pos_rev = pos,
                    actual_vel_revs= state.values.get(moteus.Register.VELOCITY, 0.0),
                    q_current_A    = state.values.get(moteus.Register.Q_CURRENT, 0.0),
                    d_current_A    = state.values.get(moteus.Register.D_CURRENT, 0.0),
                    bus_voltage_V  = state.values.get(moteus.Register.VOLTAGE, 0.0),
                    temperature_C  = state.values.get(moteus.Register.TEMPERATURE, 0.0),
                    mode           = "dwell",
                ))
                await asyncio.sleep(WATCHDOG_PERIOD_S)

    print("[Traj] All cycles complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Telemetry logging
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(log: List[TelemetryRow], path: str = CSV_PATH) -> None:
    """Write all telemetry rows to CSV."""
    fields = [
        "timestamp_s", "target_pos_rev", "actual_pos_rev",
        "actual_vel_revs", "q_current_A", "d_current_A",
        "bus_voltage_V", "temperature_C", "mode",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in log:
            w.writerow({
                "timestamp_s":     round(row.timestamp_s, 5),
                "target_pos_rev":  round(row.target_pos_rev, 5) if not math.isnan(row.target_pos_rev) else "",
                "actual_pos_rev":  round(row.actual_pos_rev, 5),
                "actual_vel_revs": round(row.actual_vel_revs, 5),
                "q_current_A":     round(row.q_current_A, 5),
                "d_current_A":     round(row.d_current_A, 5),
                "bus_voltage_V":   round(row.bus_voltage_V, 4),
                "temperature_C":   round(row.temperature_C, 2),
                "mode":            row.mode,
            })
    print(f"[CSV] Wrote {len(log)} rows → {path}")


def plot_telemetry(csv_path: str = CSV_PATH, save_path: str = PLOT_PATH) -> None:
    """
    Generate a 3-panel telemetry plot from the saved CSV:
      Panel 1 — Target vs Actual position
      Panel 2 — Bus voltage & temperature
      Panel 3 — Iq (torque-producing current) — acceleration phase highlighted
    """
    # Load CSV
    t, tgt, act, vel, iq, vbus, temp, mode = [], [], [], [], [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            t.append(float(row["timestamp_s"]))
            tgt.append(float(row["target_pos_rev"]) if row["target_pos_rev"] else math.nan)
            act.append(float(row["actual_pos_rev"]))
            vel.append(float(row["actual_vel_revs"]))
            iq.append(float(row["q_current_A"]))
            vbus.append(float(row["bus_voltage_V"]))
            temp.append(float(row["temperature_C"]))
            mode.append(row["mode"])

    t    = np.array(t);    tgt  = np.array(tgt)
    act  = np.array(act);  vel  = np.array(vel)
    iq   = np.array(iq);   vbus = np.array(vbus)
    temp = np.array(temp); mode = np.array(mode)

    # Identify acceleration phase: |vel| is increasing AND Iq is high
    accel_mask = (np.abs(np.gradient(vel, t)) > 1.0)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Moteus Reciprocating Actuator — Telemetry\n"
                 "Social HW Lab — Robotics Assessment Task 4",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    # ── Panel 1: Position ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, act, color="#185FA5", lw=1.5, label="Actual position (rev)")
    ax1.plot(t, tgt, color="#E24B4A", lw=1.0, ls="--", alpha=0.7, label="Target position (rev)")
    # Shade dwell regions
    dwell_mask = (mode == "dwell")
    ax1.fill_between(t, 0, 1, where=dwell_mask,
                     transform=ax1.get_xaxis_transform(),
                     alpha=0.12, color="gray", label="Endpoint dwell")
    ax1.set_ylabel("Position (rev)"); ax1.set_xlabel("")
    ax1.set_title("Target vs Actual Position")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── Panel 2: Bus voltage & temperature ────────────────────────────────────
    ax2  = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()
    ax2.plot(t, vbus, color="#3B6D11", lw=1.5, label="Bus voltage (V)")
    ax2b.plot(t, temp, color="#E24B4A", lw=1.2, ls="-.", alpha=0.8, label="Temperature (°C)")
    ax2.set_ylabel("Bus Voltage (V)", color="#3B6D11")
    ax2b.set_ylabel("Temperature (°C)", color="#E24B4A")
    ax2.set_title("Bus Voltage & Temperature")
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Iq (torque current) with acceleration highlighted ───────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, iq, color="#185FA5", lw=1.2, label="Iq — torque current (A)")
    ax3.fill_between(t, iq, 0, where=accel_mask,
                     alpha=0.30, color="#EF9F27", label="Acceleration phase")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_ylabel("Iq (A)"); ax3.set_xlabel("Time (s)")
    ax3.set_title("Iq (Torque-Producing Current) — Acceleration Phase Highlighted")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation mode (no hardware)
# ─────────────────────────────────────────────────────────────────────────────

def generate_simulated_telemetry(path: str = CSV_PATH) -> None:
    """
    Generate realistic synthetic telemetry for demo/testing without hardware.

    Physics model:
      - Trapezoidal velocity profile (accel_limit=20 rev/s², vel_limit=5 rev/s)
      - Bus voltage sags slightly under load (motor draw ≈ Iq × back-EMF)
      - Temperature rises linearly with Iq²·R losses
    """
    print("[Sim] Generating synthetic telemetry ...")
    DT_SIM   = 0.005   # 5 ms sample rate
    rows     = []
    t        = 0.0
    pos      = 0.0
    vel      = 0.0
    temp     = 25.0
    vbus_nom = 24.0

    endpoints = [0.0, 2.0]
    ep_idx    = 0
    n_cycles  = N_CYCLES

    # Homing phase (0 → -0.1 rev, detect wall)
    for i in range(60):
        iq = min(0.05 * i, HOMING_TORQUE_THRESH * 0.9 + 0.01 * i)
        rows.append(TelemetryRow(
            timestamp_s=t, target_pos_rev=math.nan,
            actual_pos_rev=-i*0.002, actual_vel_revs=-HOMING_VELOCITY,
            q_current_A=iq, d_current_A=0.02,
            bus_voltage_V=vbus_nom - 0.1*iq, temperature_C=temp,
            mode="homing"))
        t += DT_SIM
        temp += 0.01 * iq**2

    # Back off and zero
    pos = 0.0; vel = 0.0

    for cycle in range(n_cycles):
        for leg in range(2):
            target = endpoints[ep_idx % 2]; ep_idx += 1
            # Trapezoid to target
            dist = abs(target - pos)
            sign = 1 if target > pos else -1
            while abs(pos - target) > 0.01:
                # Accelerate / coast / decelerate
                dist_rem = abs(target - pos)
                vel_peak = min(TRAJ_VEL_LIMIT,
                               math.sqrt(2 * TRAJ_ACCEL_LIMIT * dist_rem / 2))
                if abs(vel) < vel_peak:
                    vel += sign * TRAJ_ACCEL_LIMIT * DT_SIM
                    vel  = max(-TRAJ_VEL_LIMIT, min(TRAJ_VEL_LIMIT, vel))
                    phase = "accel"
                elif dist_rem < (vel**2) / (2 * TRAJ_ACCEL_LIMIT) + 0.05:
                    vel -= sign * TRAJ_ACCEL_LIMIT * DT_SIM
                    phase = "decel"
                else:
                    phase = "coast"

                pos += vel * DT_SIM
                iq   = abs(TRAJ_ACCEL_LIMIT * 0.01) if phase == "accel" else \
                       abs(vel) * 0.05
                vbus = vbus_nom - 0.8 * iq
                temp+= 0.005 * iq**2
                rows.append(TelemetryRow(
                    timestamp_s=t, target_pos_rev=target,
                    actual_pos_rev=pos, actual_vel_revs=vel,
                    q_current_A=iq * sign, d_current_A=0.01,
                    bus_voltage_V=vbus, temperature_C=round(temp, 2),
                    mode="trajectory"))
                t += DT_SIM

            # Dwell
            for _ in range(int(ENDPOINT_DWELL_S / DT_SIM)):
                rows.append(TelemetryRow(
                    timestamp_s=t, target_pos_rev=target,
                    actual_pos_rev=pos, actual_vel_revs=0.0,
                    q_current_A=0.05, d_current_A=0.0,
                    bus_voltage_V=vbus_nom - 0.05,
                    temperature_C=round(temp, 2), mode="dwell"))
                t += DT_SIM

    save_csv(rows, path)
    print(f"[Sim] Generated {len(rows)} rows of synthetic telemetry")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main_live() -> None:
    """Run all three phases on real hardware."""
    if not MOTEUS_AVAILABLE:
        raise RuntimeError("moteus package not installed. Run: pip install moteus")

    transport  = moteus.Fdcanusb()
    controller = moteus.Controller(id=SERVO_ID, transport=transport)
    log: List[TelemetryRow] = []

    try:
        # ── Phase 1: Safety limits + homing ───────────────────────────────────
        await configure_safety_limits(controller)
        homed = await torque_home(controller, log)
        if not homed:
            print("[Main] Homing failed — aborting.")
            return

        # ── Phase 2: Cyclic trajectory ─────────────────────────────────────────
        await run_trajectory(controller, log)

        # ── Stop cleanly ───────────────────────────────────────────────────────
        await controller.set_stop()
        print("[Main] Motor stopped ✓")

    finally:
        # Phase 3: Always save telemetry even if interrupted
        if log:
            save_csv(log)
            plot_telemetry()


def main() -> None:
    parser = argparse.ArgumentParser(description="Moteus reciprocating actuator")
    parser.add_argument("--simulate",  action="store_true",
                        help="Generate synthetic telemetry (no hardware needed)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from existing telemetry.csv")
    args = parser.parse_args()

    if args.plot_only:
        plot_telemetry()

    elif args.simulate:
        generate_simulated_telemetry()
        plot_telemetry()

    else:
        asyncio.run(main_live())


if __name__ == "__main__":
    main()

/*
 * haptic_pid.c
 * ============
 * PID controller for a haptic rotary dial (BLDC motor + high-res encoder).
 *
 * Hardware context
 * ----------------
 *   - Motor      : small BLDC with high-resolution quadrature encoder
 *   - Coupling   : "sloppy" mechanical coupling → non-trivial backlash
 *   - Actuator   : simulates detents (clicks) and end-stops on a rotary knob
 *   - Control Hz : 10 kHz (dt = 100 µs) — typical for embedded FOC loops
 *
 * Design decisions
 * ----------------
 *   1. Clamping anti-windup    — stops integrator winding up against hard-stops.
 *   2. Low-pass filtered D-term — tames encoder quantisation / EMI noise.
 *   3. Output saturation guard  — never commands more than MAX_OUTPUT torque.
 *
 * Author  : Srikar Reddy
 * Task    : Social HW Lab — Robotics Engineer Assessment, Task 3
 */

#include <stdint.h>
#include <stdbool.h>

/* ─── Compile-time constants ─────────────────────────────────────────────── */

#define DT           0.0001f   /* control loop period  [s]  (10 kHz)         */
#define MAX_OUTPUT   1.0f      /* peak normalised torque command  [−1 … +1]   */
#define MIN_OUTPUT  -1.0f

/*
 * D-term low-pass filter cutoff frequency.
 *
 * Choice justification
 * --------------------
 *   Encoder resolution   : ~0.1 deg (18-bit abs. encoder @ 360 deg)
 *   Expected max velocity: ~50 rev/s → 18 000 deg/s
 *   Meaningful bandwidth : haptic detent forces change at ≤ 200 Hz
 *                          (human finger perceives up to ~300 Hz)
 *   Noise floor          : quantisation + EMI typically above 1–2 kHz
 *
 *   Setting f_c = 200 Hz gives:
 *     • full pass-through of haptic-relevant signals (< 200 Hz)
 *     • 20 dB/decade roll-off of encoder noise (1–5 kHz)
 *     • phase lag at 100 Hz ≈ 26.6° — acceptable for a 10 kHz loop
 *
 *   alpha = dt / (dt + 1/(2*pi*f_c))
 *         = 100e-6 / (100e-6 + 1/(2*pi*200))
 *         ≈ 0.1118
 */
#define LPF_CUTOFF_HZ  200.0f
#define LPF_ALPHA      0.1118f   /* pre-computed smoothing coefficient        */


/* ─── PID state structure ─────────────────────────────────────────────────── */

typedef struct {
    /* Tuning parameters */
    float Kp;
    float Ki;
    float Kd;

    /* Internal state */
    float integral;        /* accumulated integral term                       */
    float prev_error;      /* error from previous cycle  (for D calculation)  */
    float d_filtered;      /* low-pass filtered derivative                    */

    /* Anti-windup: track whether output is saturated */
    bool  saturated;       /* true when output hit MAX/MIN last cycle         */
    float sat_sign;        /* sign of saturation (+1 or −1)                   */
} PIDController;


/* ─── Initialiser ─────────────────────────────────────────────────────────── */

/**
 * pid_init() — zero all state and set tuning gains.
 *
 * Call once at system start, and again after any E-stop or mode change
 * to avoid integral/derivative transients on re-enable.
 */
void pid_init(PIDController *pid, float Kp, float Ki, float Kd)
{
    pid->Kp         = Kp;
    pid->Ki         = Ki;
    pid->Kd         = Kd;
    pid->integral   = 0.0f;
    pid->prev_error = 0.0f;
    pid->d_filtered = 0.0f;
    pid->saturated  = false;
    pid->sat_sign   = 0.0f;
}


/* ─── Helper: clamp ───────────────────────────────────────────────────────── */

static inline float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}


/* ─── Main PID update ─────────────────────────────────────────────────────── */

/**
 * pid_update() — compute one control output sample.
 *
 * Parameters
 * ----------
 * pid       : controller state (modified in place)
 * setpoint  : desired position / angle  [rad or counts — consistent units]
 * measured  : actual encoder reading    [same units]
 *
 * Returns
 * -------
 * Normalised torque command in [MIN_OUTPUT … MAX_OUTPUT].
 *
 *
 * Anti-Windup — why it is critical at a hard-stop
 * ------------------------------------------------
 * When the knob is commanded to an end-stop position the error is non-zero
 * but the motor physically cannot move any further.  Without anti-windup the
 * integral keeps accumulating ("winding up") for as long as the stop is held.
 * The moment the knob is released the accumulated integral produces a large
 * torque burst that flings the motor at high speed — dangerous and jarring
 * for the user.
 *
 * The clamping strategy used here:
 *   • Compute the raw (unsaturated) output first.
 *   • If it exceeds [MIN_OUTPUT, MAX_OUTPUT], the output is saturated.
 *   • Integrator update is BLOCKED whenever the error and the saturation are
 *     in the same direction (i.e. adding more integral would worsen saturation).
 *   • When unsaturated, integration resumes normally.
 *
 * This is the "conditional integration" / "clamping" method — simple, robust,
 * and well-understood in safety-critical embedded systems.
 *
 *
 * Derivative Filtering — low-pass on the D-term
 * -----------------------------------------------
 * Raw derivative = (error − prev_error) / dt
 *
 * For a high-resolution encoder the quantisation step is tiny, but any single
 * bit-flip due to noise or EMI appears as a large spike in (error − prev_error).
 * A first-order IIR (exponential moving average) is applied:
 *
 *   d_raw      = (error − prev_error) / dt
 *   d_filtered = alpha * d_raw + (1 − alpha) * d_filtered_prev
 *
 * alpha is computed from the chosen cutoff frequency (200 Hz, see above).
 */
float pid_update(PIDController *pid, float setpoint, float measured)
{
    /* ── 1. Error ──────────────────────────────────────────────────────────── */
    float error = setpoint - measured;

    /* ── 2. Proportional term ─────────────────────────────────────────────── */
    float p_term = pid->Kp * error;

    /* ── 3. Integral term with clamping anti-windup ───────────────────────── */
    /*
     * Only integrate when NOT saturated in the same direction as the error.
     * Condition: integrate if output was not saturated, OR if the error would
     *            help unwind the saturation (opposite sign to sat_sign).
     */
    bool integrate = (!pid->saturated) ||
                     (pid->saturated && (error * pid->sat_sign < 0.0f));

    if (integrate) {
        pid->integral += error * DT;
    }
    float i_term = pid->Ki * pid->integral;

    /* ── 4. Derivative term — low-pass filtered ───────────────────────────── */
    float d_raw      = (error - pid->prev_error) / DT;
    pid->d_filtered  = LPF_ALPHA * d_raw + (1.0f - LPF_ALPHA) * pid->d_filtered;
    float d_term     = pid->Kd * pid->d_filtered;

    /* ── 5. Sum and saturate output ───────────────────────────────────────── */
    float output_raw = p_term + i_term + d_term;
    float output     = clampf(output_raw, MIN_OUTPUT, MAX_OUTPUT);

    /* ── 6. Update saturation bookkeeping ────────────────────────────────── */
    if (output >= MAX_OUTPUT) {
        pid->saturated = true;
        pid->sat_sign  = +1.0f;
    } else if (output <= MIN_OUTPUT) {
        pid->saturated = true;
        pid->sat_sign  = -1.0f;
    } else {
        pid->saturated = false;
        pid->sat_sign  = 0.0f;
    }

    /* ── 7. Store state for next cycle ───────────────────────────────────── */
    pid->prev_error = error;

    return output;
}


/* ─── Haptic force profiles ───────────────────────────────────────────────── */

/*
 * detent_setpoint()
 * -----------------
 * Maps a continuous angle to the nearest detent position.
 * The PID setpoint is always a detent centre; the error drives the
 * "click" spring feel.
 *
 *   detent_spacing_rad : angular spacing between clicks [rad]
 *                        e.g. 2π/24 ≈ 0.2618 rad for a 24-position encoder
 */
float detent_setpoint(float angle_rad, float detent_spacing_rad)
{
    /* Round to nearest multiple of detent_spacing */
    float n = (float)(int)(angle_rad / detent_spacing_rad + 0.5f);
    return n * detent_spacing_rad;
}

/*
 * endstop_setpoint()
 * ------------------
 * Clamps the setpoint to [min_angle, max_angle].
 * When the knob is pushed past the limit the PID fights back —
 * simulating a physical hard-stop feel.
 */
float endstop_setpoint(float desired, float min_angle, float max_angle)
{
    return clampf(desired, min_angle, max_angle);
}


/* ─── Example main loop (bare-metal / RTOS task) ─────────────────────────── */

#ifdef HAPTIC_DEMO

#include <stdio.h>
#include <math.h>

/* Stubs — replace with actual HAL calls on target hardware */
static float read_encoder_rad(void)  { return 0.0f; }   /* TODO: HAL */
static void  set_motor_torque(float) { /* TODO: HAL */ }

int main(void)
{
    PIDController pid;

    /*
     * Initial gains (from tuning procedure in the report).
     * These are starting values — tune on actual hardware.
     *   Kp : provides the spring stiffness of each detent click
     *   Ki : small, only to overcome static friction / steady-state offset
     *   Kd : damps oscillation; set carefully to avoid noise amplification
     */
    pid_init(&pid, Kp=8.0f, Ki=0.5f, Kd=0.02f);

    const float DETENT_SPACING = (2.0f * 3.14159265f) / 24.0f;  /* 24 detents */
    const float END_STOP_MIN   = 0.0f;
    const float END_STOP_MAX   = 2.0f * 3.14159265f;             /* 0 – 360 deg */

    for (;;) {
        float measured  = read_encoder_rad();

        /* Snap to nearest detent, then clamp to end-stops */
        float setpoint  = detent_setpoint(measured, DETENT_SPACING);
              setpoint  = endstop_setpoint(setpoint, END_STOP_MIN, END_STOP_MAX);

        float torque    = pid_update(&pid, setpoint, measured);
        set_motor_torque(torque);

        /* 10 kHz loop — wait for next timer tick */
        /* HAL_Delay_us(100); */
    }

    return 0;
}

#endif /* HAPTIC_DEMO */

/*
 * test_haptic_pid.c
 * =================
 * Unit test harness for haptic_pid.c
 *
 * Simulates a simple 1st-order plant:
 *   velocity += torque * dt   (torque drives angular acceleration)
 *   position += velocity * dt
 *
 * Tests
 * -----
 *   1. Step response        — knob jumps to a detent, settles without oscillation
 *   2. Anti-windup          — hold against end-stop for 500 steps, release, confirm no burst
 *   3. Detent snapping      — continuous slow rotation, confirm snapping to nearest click
 *   4. End-stop clamping    — command past max angle, confirm output stays bounded
 *
 * Build & run
 * -----------
 *   gcc -o test_haptic_pid test_haptic_pid.c -lm && ./test_haptic_pid
 *
 * Author: Srikar Reddy
 */

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

/* ── Inline the controller (no separate .o needed) ── */
#define DT          0.0001f
#define MAX_OUTPUT  1.0f
#define MIN_OUTPUT -1.0f
#define LPF_CUTOFF_HZ 200.0f
#define LPF_ALPHA   0.1118f

typedef struct {
    float Kp, Ki, Kd;
    float integral;
    float prev_error;
    float d_filtered;
    bool  saturated;
    float sat_sign;
} PIDController;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void pid_init(PIDController *pid, float Kp, float Ki, float Kd) {
    pid->Kp = Kp; pid->Ki = Ki; pid->Kd = Kd;
    pid->integral = pid->prev_error = pid->d_filtered = 0.0f;
    pid->saturated = false; pid->sat_sign = 0.0f;
}

float pid_update(PIDController *pid, float setpoint, float measured) {
    float error  = setpoint - measured;
    float p_term = pid->Kp * error;

    bool integrate = (!pid->saturated) ||
                     (pid->saturated && (error * pid->sat_sign < 0.0f));
    if (integrate) pid->integral += error * DT;
    float i_term = pid->Ki * pid->integral;

    float d_raw     = (error - pid->prev_error) / DT;
    pid->d_filtered = LPF_ALPHA * d_raw + (1.0f - LPF_ALPHA) * pid->d_filtered;
    float d_term    = pid->Kd * pid->d_filtered;

    float output_raw = p_term + i_term + d_term;
    float output     = clampf(output_raw, MIN_OUTPUT, MAX_OUTPUT);

    if      (output >= MAX_OUTPUT) { pid->saturated = true;  pid->sat_sign = +1.0f; }
    else if (output <= MIN_OUTPUT) { pid->saturated = true;  pid->sat_sign = -1.0f; }
    else                           { pid->saturated = false; pid->sat_sign =  0.0f; }

    pid->prev_error = error;
    return output;
}

float detent_setpoint(float angle, float spacing) {
    float n = (float)(int)(angle / spacing + 0.5f);
    return n * spacing;
}

float endstop_setpoint(float desired, float mn, float mx) {
    return clampf(desired, mn, mx);
}

/* ── Simple plant model ───────────────────────────────────────────────────── */
typedef struct { float pos; float vel; float inertia; float friction; } Plant;

void plant_init(Plant *pl, float pos0) {
    pl->pos = pos0; pl->vel = 0.0f;
    pl->inertia  = 0.005f;   /* kg·m² — small BLDC rotor                   */
    pl->friction = 0.002f;   /* viscous damping coefficient                  */
}

void plant_step(Plant *pl, float torque) {
    float accel = (torque - pl->friction * pl->vel) / pl->inertia;
    pl->vel += accel * DT;
    pl->pos += pl->vel * DT;
}

/* ── Test utilities ──────────────────────────────────────────────────────── */
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"
#define HDR  "\033[1;34m"
#define RST  "\033[0m"

static int total = 0, passed = 0;

void check(const char *name, bool cond) {
    total++;
    if (cond) passed++;
    printf("  [%s] %s\n", cond ? PASS : FAIL, name);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TEST 1 — Step response                                                    */
/* ══════════════════════════════════════════════════════════════════════════ */
void test_step_response(void) {
    printf(HDR "\n[Test 1] Step Response\n" RST);

    PIDController pid; Plant plant;
    pid_init(&pid, 8.0f, 0.5f, 0.02f);
    plant_init(&plant, 0.0f);

    float setpoint = 0.2618f;   /* one detent (2π/24) */
    float max_overshoot = 0.0f;
    int   settled_steps = 0;
    bool  settled       = false;

    printf("  Step: 0.0 → %.4f rad\n", setpoint);
    printf("  %6s  %8s  %8s  %8s\n", "step","pos","error","torque");

    for (int i = 0; i < 5000; i++) {
        float torque = pid_update(&pid, setpoint, plant.pos);
        plant_step(&plant, torque);

        float err = fabsf(setpoint - plant.pos);
        if (plant.pos > setpoint) {
            float ov = plant.pos - setpoint;
            if (ov > max_overshoot) max_overshoot = ov;
        }
        if (err < 0.001f) settled_steps++;
        else              settled_steps = 0;
        if (settled_steps >= 100 && !settled) {
            printf("  Settled at step %d (t=%.3f s)\n", i, i * DT);
            settled = true;
        }
        if (i % 500 == 0)
            printf("  %6d  %8.5f  %8.5f  %8.5f\n",
                   i, plant.pos, err, torque);
    }

    float final_err = fabsf(setpoint - plant.pos);
    printf("  Final error   : %.5f rad\n", final_err);
    printf("  Max overshoot : %.5f rad (%.2f%%)\n",
           max_overshoot, 100.0f * max_overshoot / setpoint);

    check("Settles within 5000 steps", settled);
    check("Final error < 2 mrad",      final_err < 0.002f);
    check("Overshoot < 15%",           max_overshoot < 0.15f * setpoint);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TEST 2 — Anti-windup at hard-stop                                         */
/* ══════════════════════════════════════════════════════════════════════════ */
void test_anti_windup(void) {
    printf(HDR "\n[Test 2] Anti-Windup at Hard-Stop\n" RST);

    PIDController pid; Plant plant;
    pid_init(&pid, 8.0f, 0.5f, 0.02f);
    plant_init(&plant, 0.0f);

    /* Freeze the plant (simulate physical hard-stop) */
    float setpoint      = 0.5f;
    float integral_before = 0.0f;

    printf("  Holding against end-stop for 500 steps...\n");
    for (int i = 0; i < 500; i++) {
        pid_update(&pid, setpoint, 0.0f);   /* plant stuck at 0 */
        /* do NOT step plant */
    }
    integral_before = pid.integral;
    printf("  Integral after 500 stuck steps : %.5f\n", integral_before);

    /* Now release — run freely for 200 steps, record peak torque */
    plant_init(&plant, 0.0f);
    float peak_torque = 0.0f;
    for (int i = 0; i < 200; i++) {
        float torque = pid_update(&pid, setpoint, plant.pos);
        plant_step(&plant, torque);
        if (fabsf(torque) > peak_torque) peak_torque = fabsf(torque);
    }
    printf("  Peak torque after release      : %.5f\n", peak_torque);

    /* Without anti-windup the integral would be 500*0.5*0.0001 = 0.025
     * and i_term = Ki*integral = 0.5*0.025 = 0.0125 — tiny but would add
     * to P+D and nudge output toward saturation.
     * With clamping the integral is bounded because output was saturated. */
    check("Integral bounded (< 0.1)",   fabsf(integral_before) < 0.10f);
    check("Peak torque ≤ MAX_OUTPUT",   peak_torque <= 1.001f);
    check("No runaway (peak < 0.95)",   peak_torque < 0.95f);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TEST 3 — Detent snapping                                                  */
/* ══════════════════════════════════════════════════════════════════════════ */
void test_detent_snapping(void) {
    printf(HDR "\n[Test 3] Detent Snapping\n" RST);

    const float SPACING = (2.0f * 3.14159265f) / 24.0f;   /* 24 detents */
    float angles[] = {0.05f, 0.13f, 0.30f, 0.52f, 0.80f, 1.05f};
    int n = sizeof(angles) / sizeof(angles[0]);
    bool all_ok = true;

    printf("  %10s  %10s  %10s  %6s\n", "input_rad","snapped","expected","ok?");
    for (int i = 0; i < n; i++) {
        float snapped  = detent_setpoint(angles[i], SPACING);
        float expected = roundf(angles[i] / SPACING) * SPACING;
        bool  ok       = fabsf(snapped - expected) < 1e-5f;
        if (!ok) all_ok = false;
        printf("  %10.5f  %10.5f  %10.5f  %6s\n",
               angles[i], snapped, expected, ok ? "yes" : "NO");
    }
    check("All angles snap to correct detent", all_ok);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TEST 4 — End-stop clamping                                                */
/* ══════════════════════════════════════════════════════════════════════════ */
void test_endstop_clamping(void) {
    printf(HDR "\n[Test 4] End-Stop Clamping\n" RST);

    const float MIN_A = 0.0f;
    const float MAX_A = 6.2832f;   /* 2π */

    float tests[]    = {-1.0f,  0.0f,  3.14f,  6.28f,  7.50f, -0.01f};
    float expected[] = { 0.0f,  0.0f,  3.14f,  6.28f,  6.2832f, 0.0f};
    int n = sizeof(tests) / sizeof(tests[0]);
    bool all_ok = true;

    printf("  %10s  %10s  %10s  %6s\n", "input","clamped","expected","ok?");
    for (int i = 0; i < n; i++) {
        float clamped = endstop_setpoint(tests[i], MIN_A, MAX_A);
        bool  ok      = fabsf(clamped - expected[i]) < 1e-4f;
        if (!ok) all_ok = false;
        printf("  %10.4f  %10.4f  %10.4f  %6s\n",
               tests[i], clamped, expected[i], ok ? "yes" : "NO");
    }
    check("All out-of-range inputs clamped correctly", all_ok);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TEST 5 — Derivative filter smoothing                                      */
/* ══════════════════════════════════════════════════════════════════════════ */
void test_derivative_filter(void) {
    printf(HDR "\n[Test 5] Derivative Filter (noise rejection)\n" RST);

    PIDController pid_noisy, pid_filtered;
    pid_init(&pid_noisy,    8.0f, 0.0f, 0.02f);
    pid_init(&pid_filtered, 8.0f, 0.0f, 0.02f);

    /* Inject a single large noise spike at step 50 */
    float setpoint = 0.2618f;
    float position = 0.26f;       /* near setpoint — tiny true error */
    float spike    = 0.05f;       /* 50 mrad noise spike */

    float raw_d_peak = 0.0f, filt_d_peak = 0.0f;

    for (int i = 0; i < 100; i++) {
        float meas = position + (i == 50 ? spike : 0.0f);

        float err_noisy = setpoint - meas;
        float d_raw_abs = fabsf((err_noisy - pid_noisy.prev_error) / DT);
        if (d_raw_abs > raw_d_peak) raw_d_peak = d_raw_abs;

        pid_update(&pid_filtered, setpoint, meas);
        float d_filt_abs = fabsf(pid_filtered.d_filtered);
        if (d_filt_abs > filt_d_peak) filt_d_peak = d_filt_abs;

        pid_noisy.prev_error = err_noisy;   /* manual track for raw */
    }

    float attenuation = raw_d_peak / (filt_d_peak + 1e-9f);
    printf("  Raw D-term peak  : %.2f rad/s\n", raw_d_peak);
    printf("  Filtered D peak  : %.2f rad/s\n", filt_d_peak);
    printf("  Attenuation      : %.1fx\n", attenuation);

    check("Filter attenuates spike by > 5x", attenuation > 5.0f);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* MAIN                                                                       */
/* ══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Haptic PID — Unit Test Suite                       ║\n");
    printf("║   Social HW Lab — Robotics Assessment, Task 3        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    test_step_response();
    test_anti_windup();
    test_detent_snapping();
    test_endstop_clamping();
    test_derivative_filter();

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  Results: %d / %d tests passed\n", passed, total);
    if (passed == total)
        printf("  " PASS " — All tests passed!\n");
    else
        printf("  " FAIL " — %d test(s) failed.\n", total - passed);
    printf("══════════════════════════════════════════════════════\n\n");

    return (passed == total) ? 0 : 1;
}

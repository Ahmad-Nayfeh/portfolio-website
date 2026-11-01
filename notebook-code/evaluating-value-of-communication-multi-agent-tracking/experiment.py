import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_simulation():
    np.random.seed(42)
    out_dir = os.environ.get('OUT_DIR', '.')
    os.makedirs(out_dir, exist_ok=True)

    # Simulation parameters
    dt = 0.5
    n_steps = 100
    n_trials = 100
    q_val = 0.1  # continuous process noise intensity
    
    # Constant-velocity 2D kinematics state: [x, vx, y, vy]^T
    F = np.array([
        [1.0, dt,  0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, dt ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Process noise covariance Q
    q1 = (dt**3)/3.0 * q_val
    q2 = (dt**2)/2.0 * q_val
    q3 = dt * q_val
    Q = np.array([
        [q1,  q2,  0.0, 0.0],
        [q2,  q3,  0.0, 0.0],
        [0.0, 0.0, q1,  q2 ],
        [0.0, 0.0, q2,  q3 ]
    ])

    # Sensor locations
    sensor_A = np.array([-50.0, 0.0])
    sensor_B = np.array([50.0, 0.0])
    
    # Measurement noise: range (m) and bearing (rad)
    sigma_r_A, sigma_th_A = 3.0, np.deg2rad(2.5)
    sigma_r_B, sigma_th_B = 3.0, np.deg2rad(2.5)
    R_A = np.diag([sigma_r_A**2, sigma_th_A**2])
    R_B = np.diag([sigma_r_B**2, sigma_th_B**2])

    def h_meas(state, sensor_pos):
        dx = state[0] - sensor_pos[0]
        dy = state[2] - sensor_pos[1]
        r = np.sqrt(dx**2 + dy**2) + 1e-9
        th = np.arctan2(dy, dx)
        return np.array([r, th])

    def H_jac(state, sensor_pos):
        dx = state[0] - sensor_pos[0]
        dy = state[2] - sensor_pos[1]
        r2 = dx**2 + dy**2 + 1e-9
        r = np.sqrt(r2)
        return np.array([
            [dx/r,  0.0, dy/r,  0.0],
            [-dy/r2, 0.0, dx/r2, 0.0]
        ])

    # Quantization levels to test (bits per state dimension)
    # 0 means local agent tracking only (no communication)
    bit_rates = [0, 1, 2, 3, 4, 6, 8, -1]  # -1 represents unquantized (full precision)
    
    # Prior bounding interval for dynamic uniform quantization
    # Bounds on states: position +/- 120m, velocity +/- 15 m/s
    bound_pos = 120.0
    bound_vel = 15.0
    state_bounds = np.array([bound_pos, bound_vel, bound_pos, bound_vel])

    def quantize_state(x, bits):
        if bits <= 0:
            return x.copy(), np.zeros(4)
        # Dynamic range: [-bound, bound]
        L = 2**bits
        delta = 2.0 * state_bounds / L
        # Clamp and uniform quantize
        x_clamped = np.clip(x, -state_bounds + 1e-6, state_bounds - 1e-6)
        idx = np.floor((x_clamped + state_bounds) / delta)
        idx = np.clip(idx, 0, L - 1)
        x_q = -state_bounds + (idx + 0.5) * delta
        # Quantization noise variance per dimension: delta^2 / 12
        var_q = (delta**2) / 12.0
        return x_q, var_q

    def covariance_intersection(x1, P1, x2, P2):
        inv_P1 = np.linalg.pinv(P1)
        inv_P2 = np.linalg.pinv(P2)
        best_tr = float('inf')
        best_w = 0.5
        best_P = P1
        best_x = x1
        # 1D grid search over omega in [0, 1]
        for w in np.linspace(0.0, 1.0, 21):
            inv_P = w * inv_P1 + (1.0 - w) * inv_P2
            P = np.linalg.pinv(inv_P)
            tr = np.trace(P)
            if tr < best_tr:
                best_tr = tr
                best_w = w
                best_P = P
        best_x = best_P @ (best_w * inv_P1 @ x1 + (1.0 - best_w) * inv_P2 @ x2)
        return best_x, best_P

    pos_errors = {b: [] for b in bit_rates}
    vel_errors = {b: [] for b in bit_rates}
    anees_hist = {b: np.zeros(n_steps) for b in [0, 2, 4, -1]}

    # For trajectory figure storage
    sample_truth = None
    sample_local_A = None
    sample_fused_2bit = None
    sample_fused_unq = None

    for trial in range(n_trials):
        # Initial state: start near (-30, 2, -20, 2)
        true_state = np.array([-30.0 + np.random.randn()*2.0, 
                               2.0 + np.random.randn()*0.5,
                               -20.0 + np.random.randn()*2.0,
                               2.5 + np.random.randn()*0.5])
        
        # Initialize filters at Agent A and Agent B
        init_cov = np.diag([10.0**2, 2.0**2, 10.0**2, 2.0**2])
        xA_f = true_state + np.random.multivariate_normal(np.zeros(4), init_cov)
        PA_f = init_cov.copy()
        xB_f = true_state + np.random.multivariate_normal(np.zeros(4), init_cov)
        PB_f = init_cov.copy()

        # Keep filter instances for each bit rate
        # Local states for each scheme
        filters = {}
        for b in bit_rates:
            filters[b] = {'xA': xA_f.copy(), 'PA': PA_f.copy(), 'xB': xB_f.copy(), 'PB': PB_f.copy()}

        trial_pos_err = {b: [] for b in bit_rates}
        trial_vel_err = {b: [] for b in bit_rates}
        
        if trial == 0:
            sample_truth = []
            sample_local_A = []
            sample_fused_2bit = []
            sample_fused_unq = []

        for k in range(n_steps):
            # Propagate true state
            w_proc = np.random.multivariate_normal(np.zeros(4), Q)
            true_state = F @ true_state + w_proc

            # Generate noisy measurements
            z_A_true = h_meas(true_state, sensor_A)
            v_A = np.random.multivariate_normal(np.zeros(2), R_A)
            z_A = z_A_true + v_A
            z_A[1] = np.arctan2(np.sin(z_A[1]), np.cos(z_A[1]))

            z_B_true = h_meas(true_state, sensor_B)
            v_B = np.random.multivariate_normal(np.zeros(2), R_B)
            z_B = z_B_true + v_B
            z_B[1] = np.arctan2(np.sin(z_B[1]), np.cos(z_B[1]))

            if trial == 0:
                sample_truth.append((true_state[0], true_state[2]))

            # Update each bit rate scheme
            for b in bit_rates:
                xA = filters[b]['xA']
                PA = filters[b]['PA']
                xB = filters[b]['xB']
                PB = filters[b]['PB']

                # 1. EKF Prediction
                xA_pred = F @ xA
                PA_pred = F @ PA @ F.T + Q
                xB_pred = F @ xB
                PB_pred = F @ PB @ F.T + Q

                # 2. Local EKF Measurement Update for Agent A
                HA = H_jac(xA_pred, sensor_A)
                innov_A = z_A - h_meas(xA_pred, sensor_A)
                innov_A[1] = np.arctan2(np.sin(innov_A[1]), np.cos(innov_A[1]))
                SA = HA @ PA_pred @ HA.T + R_A
                KA = PA_pred @ HA.T @ np.linalg.pinv(SA)
                xA_post = xA_pred + KA @ innov_A
                PA_post = (np.eye(4) - KA @ HA) @ PA_pred

                # 3. Local EKF Measurement Update for Agent B
                HB = H_jac(xB_pred, sensor_B)
                innov_B = z_B - h_meas(xB_pred, sensor_B)
                innov_B[1] = np.arctan2(np.sin(innov_B[1]), np.cos(innov_B[1]))
                SB = HB @ PB_pred @ HB.T + R_B
                KB = PB_pred @ HB.T @ np.linalg.pinv(SB)
                xB_post = xB_pred + KB @ innov_B
                PB_post = (np.eye(4) - KB @ HB) @ PB_pred

                # 4. Quantized Belief Sharing and Covariance Intersection Fusion
                if b == 0:
                    # Local tracking only (Agent A does not communicate with Agent B)
                    x_fused = xA_post
                    P_fused = PA_post
                elif b == -1:
                    # Ideal unquantized CI fusion
                    x_fused, P_fused = covariance_intersection(xA_post, PA_post, xB_post, PB_post)
                    # Mutual exchange
                    xB_fused, PB_fused = covariance_intersection(xB_post, PB_post, xA_post, PA_post)
                    filters[b]['xB'] = xB_fused
                    filters[b]['PB'] = PB_fused
                else:
                    # Quantize Agent B's posterior and inflate covariance by quantization noise
                    xB_q, var_q_B = quantize_state(xB_post, b)
                    PB_inflated = PB_post + np.diag(var_q_B)

                    # Quantize Agent A's posterior for Agent B
                    xA_q, var_q_A = quantize_state(xA_post, b)
                    PA_inflated = PA_post + np.diag(var_q_A)

                    # Covariance Intersection fusion at Agent A
                    x_fused, P_fused = covariance_intersection(xA_post, PA_post, xB_q, PB_inflated)
                    # Covariance Intersection fusion at Agent B
                    xB_fused, PB_fused = covariance_intersection(xB_post, PB_post, xA_q, PA_inflated)
                    filters[b]['xB'] = xB_fused
                    filters[b]['PB'] = PB_fused

                filters[b]['xA'] = x_fused
                filters[b]['PA'] = P_fused

                # Tracking error metrics at Agent A
                pos_err = np.sqrt((x_fused[0] - true_state[0])**2 + (x_fused[2] - true_state[2])**2)
                vel_err = np.sqrt((x_fused[1] - true_state[1])**2 + (x_fused[3] - true_state[3])**2)
                trial_pos_err[b].append(pos_err)
                trial_vel_err[b].append(vel_err)

                if b in anees_hist:
                    err = x_fused - true_state
                    anees = float(err.T @ np.linalg.pinv(P_fused) @ err)
                    anees_hist[b][k] += anees

                if trial == 0:
                    if b == 0:
                        sample_local_A.append((x_fused[0], x_fused[2]))
                    elif b == 2:
                        sample_fused_2bit.append((x_fused[0], x_fused[2]))
                    elif b == -1:
                        sample_fused_unq.append((x_fused[0], x_fused[2]))

        for b in bit_rates:
            pos_errors[b].append(np.mean(trial_pos_err[b]))
            vel_errors[b].append(np.mean(trial_vel_err[b]))

    # Average ANEES across trials
    for b in anees_hist:
        anees_hist[b] /= n_trials

    # Compute summary statistics
    summary_results = {}
    mean_pos_rmse = {}
    mean_vel_rmse = {}
    for b in bit_rates:
        b_key = "local_0bit" if b == 0 else ("unquantized" if b == -1 else f"{b}bit")
        mean_pos_rmse[b_key] = float(np.mean(pos_errors[b]))
        mean_vel_rmse[b_key] = float(np.mean(vel_errors[b]))

    # Fusion gain calculation relative to local baseline and unquantized upper bound
    local_pos = mean_pos_rmse["local_0bit"]
    unq_pos = mean_pos_rmse["unquantized"]
    max_possible_gain = local_pos - unq_pos

    fusion_gains_pct = {}
    for b in bit_rates:
        b_key = "local_0bit" if b == 0 else ("unquantized" if b == -1 else f"{b}bit")
        gain_m = local_pos - mean_pos_rmse[b_key]
        pct = (gain_m / max_possible_gain) * 100.0 if max_possible_gain > 0 else 0.0
        fusion_gains_pct[b_key] = float(pct)

    summary_results["mean_pos_rmse_m"] = mean_pos_rmse
    summary_results["mean_vel_rmse_mps"] = mean_vel_rmse
    summary_results["gain_pct_of_ideal"] = fusion_gains_pct
    summary_results["overall_max_gain_meters"] = float(max_possible_gain)
    summary_results["anees_mean_last_20_steps"] = {
        b_key: float(np.mean(anees_hist[b][-20:]))
        for b, b_key in [(0, "local_0bit"), (2, "2bit"), (4, "4bit"), (-1, "unquantized")]
    }

    with open("results.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    # --- FIGURE 1: RMSE vs Bit Budget ---
    bit_labels = ['0 (local)', '1', '2', '3', '4', '6', '8', 'Unquant.']
    pos_vals = [mean_pos_rmse['local_0bit'], mean_pos_rmse['1bit'], mean_pos_rmse['2bit'],
                mean_pos_rmse['3bit'], mean_pos_rmse['4bit'], mean_pos_rmse['6bit'],
                mean_pos_rmse['8bit'], mean_pos_rmse['unquantized']]
    vel_vals = [mean_vel_rmse['local_0bit'], mean_vel_rmse['1bit'], mean_vel_rmse['2bit'],
                mean_vel_rmse['3bit'], mean_vel_rmse['4bit'], mean_vel_rmse['6bit'],
                mean_vel_rmse['8bit'], mean_vel_rmse['unquantized']]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color = 'tab:blue'
    ax1.set_xlabel('Quantization Bit Rate (bits / dimension)', fontsize=11)
    ax1.set_ylabel('Position RMSE (m)', color=color, fontsize=11)
    l1 = ax1.plot(bit_labels, pos_vals, 'o-', color=color, linewidth=2, label='Position RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Velocity RMSE (m/s)', color=color, fontsize=11)
    l2 = ax2.plot(bit_labels, vel_vals, 's--', color=color, linewidth=2, label='Velocity RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Tracking Error vs. Communication Rate in CI Multi-Agent Tracking', fontsize=12)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_vs_bits.png'), dpi=150)
    plt.close()

    # --- FIGURE 2: Value of Communication Curve (Gain Knee) ---
    gains = [fusion_gains_pct['local_0bit'], fusion_gains_pct['1bit'], fusion_gains_pct['2bit'],
             fusion_gains_pct['3bit'], fusion_gains_pct['4bit'], fusion_gains_pct['6bit'],
             fusion_gains_pct['8bit'], 100.0]
    
    plt.figure(figsize=(7, 4))
    plt.plot(bit_labels, gains, 'd-', color='#2ca02c', linewidth=2.2, markersize=7)
    plt.axhline(90.0, color='gray', linestyle=':', label='90% Gain Threshold')
    plt.xlabel('Quantization Bit Rate (bits / dimension)', fontsize=11)
    plt.ylabel('Tracking Gain Achieved (% of Ideal Fusion)', fontsize=11)
    plt.title('Value of Communication (VoC): Cooperative Fusion Gain', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fusion_gain_knee.png'), dpi=150)
    plt.close()

    # --- FIGURE 3: Sample 2D Tracking Trajectory ---
    sample_truth = np.array(sample_truth)
    sample_local_A = np.array(sample_local_A)
    sample_fused_2bit = np.array(sample_fused_2bit)
    sample_fused_unq = np.array(sample_fused_unq)

    plt.figure(figsize=(7, 5))
    plt.plot(sample_truth[:, 0], sample_truth[:, 1], 'k-', linewidth=2.5, label='True Trajectory')
    plt.plot(sample_local_A[:, 0], sample_local_A[:, 1], 'r:', alpha=0.75, label='Local Filter (0 bits)')
    plt.plot(sample_fused_2bit[:, 0], sample_fused_2bit[:, 1], 'b--', alpha=0.85, label='Quantized CI (2 bits)')
    plt.plot(sample_fused_unq[:, 0], sample_fused_unq[:, 1], 'g-.', alpha=0.9, label='Ideal CI (Unquantized)')
    plt.plot(sensor_A[0], sensor_A[1], 'm^', markersize=9, label='Agent A')
    plt.plot(sensor_B[0], sensor_B[1], 'cv', markersize=9, label='Agent B')
    plt.xlabel('X Position (m)', fontsize=11)
    plt.ylabel('Y Position (m)', fontsize=11)
    plt.title('Estimated Trajectory Under Finite-Rate Information Exchange', fontsize=12)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sample_trajectory_tracking.png'), dpi=150)
    plt.close()

    # --- FIGURE 4: ANEES Consistency Analysis ---
    time_steps = np.arange(n_steps) * dt
    plt.figure(figsize=(7, 4))
    plt.plot(time_steps, anees_hist[0], label='Local Filter (0 bits)', color='red', alpha=0.8)
    plt.plot(time_steps, anees_hist[2], label='Quantized CI (2 bits)', color='blue', alpha=0.8)
    plt.plot(time_steps, anees_hist[4], label='Quantized CI (4 bits)', color='purple', alpha=0.8)
    plt.plot(time_steps, anees_hist[-1], label='Ideal CI (Unquantized)', color='green', alpha=0.8)
    plt.axhline(4.0, color='black', linestyle='--', label='Expected 4D ANEES (dim=4)')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Average NEES (ANEES)', fontsize=11)
    plt.title('Filter Consistency Across Communication Rates', fontsize=12)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'anees_consistency.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    run_simulation()

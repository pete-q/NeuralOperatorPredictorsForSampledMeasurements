import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.simulate import build_robot, make_reference, make_simulator, sample_initial_state

def exact_predictor_label(sim, cfg, q_meas, v_meas, u_hist):
    q_pred, v_pred, _ = sim["approximate_predictor"](
        q_meas,
        v_meas,
        u_hist,
        h=cfg["h_pred"],
        tol=1e-10,
        max_iters=100,
        M=8,
    )
    return q_pred, v_pred

def extract_predictor_samples(out, sim, cfg, stride=1, rollout_id=None, verbose=False):

    q_meas = out["q_meas"]
    v_meas = out["v_meas"]
    u_hist = out["u_hist"]

    n_steps = len(q_meas)
    n_samples_est = (n_steps + stride - 1) // stride

    if verbose:
        prefix = f"[rollout {rollout_id:02d}] " if rollout_id is not None else ""
        print(
            f"{prefix}extract_predictor_samples start | "
            f"steps={n_steps} stride={stride} "
            f"(~{n_samples_est} samples expected)"
        )

    X = []
    U = []
    Y = []

    start = time.perf_counter()

    for i, k in enumerate(range(0, len(q_meas), stride)):

        q_input = q_meas[k]
        v_input = v_meas[k]
        u_input = u_hist[k]

        q_pred, v_pred = exact_predictor_label(
            sim,
            cfg,
            q_input,
            v_input,
            u_input,
        )

        X.append(np.concatenate([q_input, v_input]))
        U.append(u_input)
        Y.append(np.concatenate([q_pred, v_pred]))

        # progress logging every 500 samples
        if verbose and (i + 1) % 100 == 0:
            prefix = f"[rollout {rollout_id:02d}] " if rollout_id is not None else ""
            print(
                f"{prefix}extraction progress: "
                f"{i+1}/{n_samples_est} samples"
            )

    elapsed = time.perf_counter() - start

    if verbose:
        prefix = f"[rollout {rollout_id:02d}] " if rollout_id is not None else ""
        print(
            f"{prefix}extract_predictor_samples done | "
            f"samples={len(X)} | "
            f"time={elapsed:.2f}s"
        )

    return np.array(X), np.array(U), np.array(Y)

def validate_dataset_shapes(dataset, robot, cfg):
    nq = robot["nq"]
    nv = robot["nv"]
    delay_steps = cfg["delay_steps"]

    X = dataset["state"]
    U = dataset["u_hist"]
    Y = dataset["predictor"]

    print("state shape     :", X.shape)
    print("u_hist shape    :", U.shape)
    print("predictor shape :", Y.shape)

    assert X.ndim == 2
    assert U.ndim == 3
    assert Y.ndim == 2

    assert X.shape[1] == nq + nv, f"Expected state dim {nq+nv}, got {X.shape[1]}"
    assert U.shape[1] == delay_steps, f"Expected delay_steps {delay_steps}, got {U.shape[1]}"
    assert U.shape[2] == nv, f"Expected control dim {nv}, got {U.shape[2]}"
    assert Y.shape[1] == nq + nv, f"Expected predictor dim {nq+nv}, got {Y.shape[1]}"

    assert X.shape[0] == U.shape[0] == Y.shape[0], "Mismatched number of samples"

    print("Shape checks passed.")

def validate_dataset_labels(
    dataset,
    sim,
    cfg,
    n_checks=20,
    seed=0,
    tol=1e-9,
    max_iters=100,
    M=8,
):
    rng = np.random.default_rng(seed)

    X = dataset["state"]
    U = dataset["u_hist"]
    Y = dataset["predictor"]

    nq = X.shape[1] // 2

    idxs = rng.choice(len(X), size=min(n_checks, len(X)), replace=False)

    errors = []

    for idx in idxs:
        x = X[idx]
        u_hist = U[idx]
        y_stored = Y[idx]

        q_in = x[:nq]
        v_in = x[nq:]

        q_pred, v_pred, _ = sim["approximate_predictor"](
            q_in,
            v_in,
            u_hist,
            h=cfg["h_pred"],
            tol=tol,
            max_iters=max_iters,
            M=M,
        )

        y_recomputed = np.concatenate([q_pred, v_pred])
        err = np.linalg.norm(y_recomputed - y_stored)
        errors.append(err)

    errors = np.array(errors)

    print(f"label recomputation checks: {len(errors)}")
    print(f"mean error: {errors.mean():.3e}")
    print(f"max error : {errors.max():.3e}")

    return errors
    
def save_predictor_dataset(dataset, cfg, path):

    np.savez_compressed(
        path,
        state=dataset["state"],
        u_hist=dataset["u_hist"],
        predictor=dataset["predictor"],
        config=cfg,
    )


def _run_one_rollout(args):
    """
    One rollout executed in one worker process.
    """
    (
        rollout_idx,
        rollout_seed,
        cfg,
        stride,
        q_meas_noise_std,
        v_meas_noise_std,
        use_noisy_measurement_for_reset,
    ) = args

    # Build fresh robot/reference/simulator inside this process
    robot = build_robot(cfg["urdf"])
    ref = make_reference(robot, cfg)
    sim = make_simulator(robot, cfg, ref)

    rng = np.random.default_rng(rollout_seed)
    q0, v0 = sample_initial_state(robot, rng)

    out = sim["simulate"](
        q0=q0,
        v0=v0,
        q_meas_noise_std=q_meas_noise_std,
        v_meas_noise_std=v_meas_noise_std,
        use_noisy_measurement_for_reset=use_noisy_measurement_for_reset,
        rng=rng,
        verbose=False,
        log_every_step=True,
    
        rollout_id=rollout_idx,
        progress_interval=5000,
    )

    extract_start = time.perf_counter()
    
    print(f"[rollout {rollout_idx:02d}] starting sample extraction")
    
    X, U, Y = extract_predictor_samples(
        out,
        sim,
        cfg,
        stride,
        rollout_id=rollout_idx,
        verbose=True,
    )    
    extract_elapsed = time.perf_counter() - extract_start
    
    print(
        f"[rollout {rollout_idx:02d}] finished extraction | "
        f"samples kept: {len(X)} | "
        f"extract time: {extract_elapsed:.2f}s"
    )
    return {
        "rollout_idx": rollout_idx,
        "seed": int(rollout_seed),
        "num_samples": len(X),
        "state": X,
        "u_hist": U,
        "predictor": Y,
    }


def build_predictor_dataset_parallel(
    cfg,
    n_rollouts=20,
    stride=2,
    seed=0,
    max_workers=None,
    q_meas_noise_std=0.01,
    v_meas_noise_std=0.01,
    use_noisy_measurement_for_reset=True,
    verbose=True,
):
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if verbose:
        print("\n==============================")
        print("PARALLEL DATASET GENERATION")
        print("==============================")
        print(f"rollouts        : {n_rollouts}")
        print(f"workers         : {max_workers}")
        print(f"stride          : {stride}")
        print(f"sim steps       : {cfg['steps']}")
        print(f"delay steps     : {cfg['delay_steps']}")
        print(f"sample steps    : {cfg['sample_steps']}")
        print(f"seed            : {seed}")
        print("==============================\n")

    master_rng = np.random.default_rng(seed)

    rollout_seeds = master_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=n_rollouts,
        dtype=np.uint32,
    )

    task_args = [
        (
            i,
            int(rollout_seeds[i]),
            cfg,
            stride,
            q_meas_noise_std,
            v_meas_noise_std,
            use_noisy_measurement_for_reset,
        )
        for i in range(n_rollouts)
    ]

    results = [None] * n_rollouts
    total_samples = 0

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(_run_one_rollout, arg) for arg in task_args]

        completed = 0

        for fut in as_completed(futures):

            result = fut.result()

            i = result["rollout_idx"]
            results[i] = result

            completed += 1
            samples = result["num_samples"]
            total_samples += samples

            elapsed = time.perf_counter() - start_time

            if verbose:
                print(
                    f"[{completed:>3}/{n_rollouts}] "
                    f"rollout {i:>3} finished | "
                    f"samples: {samples:>6} | "
                    f"total samples: {total_samples:>7} | "
                    f"time: {elapsed:6.2f}s"
                )

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------
    # Stack results
    # --------------------------------------------------

    nonempty = [r for r in results if r is not None and r["num_samples"] > 0]

    if len(nonempty) == 0:
        raise RuntimeError("No rollout produced any samples.")

    X_all = np.vstack([r["state"] for r in nonempty])
    U_all = np.vstack([r["u_hist"] for r in nonempty])
    Y_all = np.vstack([r["predictor"] for r in nonempty])

    samples_per_rollout = np.array(
        [0 if r is None else r["num_samples"] for r in results],
        dtype=int,
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    total_samples = X_all.shape[0]
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0

    if verbose:

        print("\n==============================")
        print("DATASET BUILD COMPLETE")
        print("==============================")
        print(f"wall time          : {elapsed:.2f} sec")
        print(f"total samples      : {total_samples}")
        print(f"samples/sec        : {samples_per_sec:.1f}")
        print("")
        print(f"state shape        : {X_all.shape}")
        print(f"u_hist shape       : {U_all.shape}")
        print(f"predictor shape    : {Y_all.shape}")
        print("")
        print("samples per rollout:")
        print(samples_per_rollout)
        print("==============================\n")

    return {
        "state": X_all,
        "u_hist": U_all,
        "predictor": Y_all,
        "rollout_seeds": rollout_seeds.copy(),
        "samples_per_rollout": samples_per_rollout,
        "config": cfg,
    }
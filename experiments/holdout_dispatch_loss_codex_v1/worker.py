"""One bounded training job. Only development data are loaded here."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import construct_model, mean_realized, read_json, state_hash, utcnow, write_json
from .loss import training_loss
from experiments.holdout_rmse_v1.sequence_models import fit_preprocess_stats, apply_preprocess


def run_job(job_path):
    job_path = Path(job_path)
    job = read_json(job_path)
    out = job_path.parent
    settings = job["training"]
    torch.set_num_threads(job["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(False)
    started, cpu_started = time.perf_counter(), time.process_time()
    with np.load(job["development_path"], allow_pickle=False) as data:
        x, y = data["x"], data["y"].astype(np.float32)
    if len(x) != 301:
        raise ValueError("worker accepts exactly 301 development days, never test labels")
    is_search = job["stage"] == "search"
    count = 271 if is_search else 301
    fill, mean, scale = fit_preprocess_stats(x[:count])
    train_x = torch.from_numpy(apply_preprocess(x[:count], fill, mean, scale))
    train_y = torch.from_numpy(y[:count])
    val_x = torch.from_numpy(apply_preprocess(x[271:], fill, mean, scale)) if is_search else None
    val_y = y[271:] if is_search else None
    model = construct_model(job["family"], x.shape[-1], job["seed"])
    initial_hash = state_hash(model.state_dict())
    # Match stochastic draws across methods after identical initialization.
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    policies = (["realized", "rmse"] if job["method"] == "mse" else ["realized"]) if is_search else []
    active = set(policies)
    best = {key: {"epoch": 0, "wait": 0, "value": None, "stop_epoch": None} for key in policies}
    max_epochs = settings["max_epochs"] if is_search else job["epochs"]
    history = []
    steps = 0

    def save_checkpoint(name, epoch):
        path = out / f"{name}.pt"
        torch.save({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "family": job["family"], "n_features": int(x.shape[-1]),
            "fill_values": fill, "mean": mean, "scale": scale,
            "epoch": epoch, "method": job["method"], "seed": job["seed"],
            "initial_state_sha256": initial_hash,
        }, path)
        return str(path)

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = np.random.permutation(count)
        running = []
        for start in range(0, count, settings["batch_size"]):
            index = permutation[start:start + settings["batch_size"]]
            prediction = model(train_x[index])
            loss = training_loss(prediction, train_y[index], job["method"], **job["params"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            steps += 1
            running.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(running)), "elapsed_seconds": time.perf_counter() - started}
        if is_search:
            model.eval()
            with torch.no_grad():
                pred = model(val_x).cpu().numpy()
            realized = mean_realized(val_y, pred)
            rmse = float(np.sqrt(np.mean((val_y.astype(float) - pred.astype(float)) ** 2)))
            row.update(val_realized=realized, val_rmse=rmse)
            for policy in list(active):
                value = realized if policy == "realized" else rmse
                old = best[policy]
                improvement = old["value"] is None or (value > old["value"] + 1e-3 if policy == "realized" else value < old["value"] - 1e-6)
                if improvement:
                    old.update(epoch=epoch, wait=0, value=value, val_realized=realized, val_rmse=rmse)
                    old["checkpoint"] = save_checkpoint(f"best_{policy}", epoch)
                    np.save(out / f"validation_{policy}.npy", pred)
                else:
                    old["wait"] += 1
                    if old["wait"] >= settings["patience"]:
                        old["stop_epoch"] = epoch
                        active.remove(policy)
            print(f"epoch={epoch:02d} loss={row['train_loss']:.5f} val_realized={realized:.3f} val_rmse={rmse:.5f}", flush=True)
        elif epoch % 10 == 0 or epoch == max_epochs:
            print(f"refit_epoch={epoch}/{max_epochs} loss={row['train_loss']:.5f}", flush=True)
        history.append(row)
        write_json(out / "progress.json", {"job_id": job["job_id"], "epoch": epoch, "history": history, "updated_utc": utcnow()})
        if is_search and not active:
            break
    checkpoint = None if is_search else save_checkpoint("final", max_epochs)
    for policy in policies:
        if best[policy]["stop_epoch"] is None:
            best[policy]["stop_epoch"] = len(history)
    result = {
        "status": "ok", "job_id": job["job_id"], "stage": job["stage"], "method": job["method"],
        "family": job["family"], "params": job["params"], "policies": best, "checkpoint": checkpoint,
        "epochs_run": len(history), "optimizer_steps": steps,
        "initial_state_sha256": initial_hash,
        "seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "completed_utc": utcnow(), "training_days": count, "holdout_labels_loaded": False,
        "history": history,
    }
    write_json(out / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    run_job(parser.parse_args().job)

"""Coordinator: bounded search/refit, then candidate freeze before testing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import (
    HERE, ROOT, PROTOCOL_PATH, Ledger, check_protected, prepare_data,
    read_json, scaled_count, sha256, utcnow, write_json,
)


def run_directory(protocol):
    return ROOT / "reports/holdout_dispatch_loss_codex_v1/runs" / protocol["run_id"]


def snapshot(run_dir, protocol, stage):
    ledger = Ledger(run_dir, protocol)
    rows = ledger.rows()
    completed = [r for r in rows if r["event"] == "training_end" and r.get("status") == "ok"]
    status = {
        "stage": stage, "run_id": protocol["run_id"], "updated_utc": utcnow(),
        "started_utc": protocol["budget_started_at_utc"], "deadline_utc": protocol["deadline_utc"],
        "training_launches": ledger.launches(), "training_launch_limit": protocol["training_launch_limit"],
        "remaining_seconds": max(0, ledger.remaining_seconds()),
        "worker_wall_seconds": sum(r.get("seconds", 0) for r in completed),
        "worker_cpu_seconds": sum(r.get("cpu_seconds", 0) for r in completed),
        "external_cost_usd": 0, "training_failures": [r for r in rows if r["event"] == "training_end" and r.get("status") != "ok"],
    }
    write_json(run_dir / "status.json", status)
    lines = [
        "# Codex 调度损失试验：权威状态", "",
        f"- 阶段：{stage}", f"- 运行：`{run_dir.relative_to(ROOT).as_posix()}`",
        f"- 开始：{status['started_utc']}；截止：{status['deadline_utc']}；恢复不重新计时。",
        f"- 已启动训练：{status['training_launches']} / {status['training_launch_limit']}；剩余墙钟 {status['remaining_seconds'] / 60:.1f} 分钟；外部费用 0。",
        f"- 工作进程累计墙钟 {status['worker_wall_seconds']:.1f} 秒、CPU {status['worker_cpu_seconds']:.1f} 秒；失败见 `ledger.jsonl`。",
        "- 主方法每模型三配置，matched CE 同等搜索，MSE 双选模规则共享轨迹、分别停止；对照成本单列。",
        "- 未把 holdout 标签传给训练器；所有候选冻结后才测试。未建立操作系统隔离，不宣称正式隔离实验。",
        "- 不修改 Cursor、原始数据、历史输出、旧 notebook；不访问封存目录。",
        "- 结论以 `comparison.csv` / `analysis.json` 为准，尚未评价时不能推测是否获胜。",
        "- 原 Cursor 构造模型前未固定种子；matched CE 控制该差异。soft-regret 与 idle 是组合改动，不分别归因。",
        "- 下一入口：`python -m experiments.holdout_dispatch_loss_codex_v1.run train` 可恢复；冻结后用 `...evaluate`，不得新增训练配置。",
    ]
    (HERE / "STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def execute_job(run_dir, protocol, spec):
    ledger = Ledger(run_dir, protocol)
    base_id = spec["job_id"]
    previous_starts = [r for r in ledger.rows() if r["event"] == "training_start" and r["base_job_id"] == base_id]
    if previous_starts:
        latest = previous_starts[-1]
        result_path = Path(latest["job_dir"]) / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            if result.get("status") == "ok":
                ledger.add("cache_hit", job_id=base_id, result=str(result_path))
                return result
        retry_count = sum(r.get("attempt", 1) > 1 for r in ledger.rows() if r["event"] == "training_start")
        if retry_count >= protocol["total_retry_limit"]:
            raise RuntimeError("retry budget exhausted")
    ledger.check_start(protocol["job_timeout_seconds"])
    attempt = len(previous_starts) + 1
    job_dir = run_dir / "jobs" / f"{base_id}_attempt{attempt}"
    job_dir.mkdir(parents=True, exist_ok=False)
    job = {
        **spec, "training": protocol["training"], "seed": protocol["seed"],
        "threads": protocol["torch_cpu_threads"],
        "development_path": str(run_dir / "data/development.npz"),
    }
    write_json(job_dir / "job.json", job)
    ledger.add("training_start", base_job_id=base_id, job_id=base_id, attempt=attempt, job_dir=str(job_dir),
               stage=spec["stage"], method=spec["method"], family=spec["family"], params=spec["params"])
    print(f"START {base_id} attempt={attempt} launches={ledger.launches()}/34", flush=True)
    snapshot(run_dir, protocol, f"TRAINING {base_id}")
    started = time.perf_counter()
    environment = dict(os.environ)
    environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    try:
        with (job_dir / "stdout.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-X", "utf8", "-B", "-m", "experiments.holdout_dispatch_loss_codex_v1.worker", str(job_dir / "job.json")],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                timeout=protocol["job_timeout_seconds"], check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        result = read_json(job_dir / "result.json")
        ledger.add("training_end", job_id=base_id, attempt=attempt, status="ok", seconds=result["seconds"],
                   cpu_seconds=result["cpu_seconds"], subprocess_wall_seconds=time.perf_counter() - started,
                   epochs=result["epochs_run"], optimizer_steps=result["optimizer_steps"], method=spec["method"], stage=spec["stage"])
        if spec["stage"] == "search":
            print("DONE", base_id, {k: (v["epoch"], round(v["val_realized"], 3)) for k, v in result["policies"].items()}, flush=True)
        else:
            print("DONE", base_id, "refit_epochs", result["epochs_run"], flush=True)
        return result
    except Exception as exc:
        ledger.add("training_end", job_id=base_id, attempt=attempt, status="failed", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - started)
        snapshot(run_dir, protocol, f"FAILED {base_id}; inspect saved stdout.log before resuming")
        raise


def freeze_protocol(run_dir, protocol):
    frozen_path = run_dir / "protocol_frozen.json"
    code = {str((HERE / name).relative_to(ROOT)): sha256(HERE / name) for name in ["__init__.py", "loss.py", "common.py", "worker.py", "run.py", "protocol.json"]}
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        if frozen["code_sha256"] != code:
            raise RuntimeError("training code changed after protocol freeze; no silent resume")
        return frozen
    frozen = {"protocol": protocol, "code_sha256": code, "frozen_utc": utcnow(), "data": read_json(run_dir / "data/manifest.json")}
    write_json(frozen_path, frozen)
    Ledger(run_dir, protocol).add("protocol_frozen", fingerprint=sha256(frozen_path))
    return frozen


def train_all(run_dir, protocol):
    if (run_dir / "candidate_freeze.json").exists():
        print("Candidates already frozen; no training restarted.")
        return
    prepare_data(run_dir)
    freeze_protocol(run_dir, protocol)
    lock_path = run_dir / "training.lock"
    if lock_path.exists():
        import psutil
        previous = read_json(lock_path)
        if psutil.pid_exists(previous["pid"]):
            raise RuntimeError(f"task coordinator still exists: pid {previous['pid']}")
        lock_path.rename(run_dir / f"training.stale.{int(time.time())}.json")
    with lock_path.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started_utc": utcnow()}, f)
    try:
        searches = []
        for method in ["soft_regret", "cursor_ce", "mse"]:
            configs = protocol["grid"] if method != "mse" else [{"lam": 0.0, "tau": 1.0}]
            for family in protocol["families"]:
                for index, params in enumerate(configs):
                    result = execute_job(run_dir, protocol, {"job_id": f"search_{method}_{family}_{index}", "stage": "search", "method": method, "family": family, "params": params})
                    searches.append(result)
        # All methods search before any test evaluation. Choices use only the
        # 30 validation days; historical test scores never enter this logic.
        candidates = []
        for method in ["soft_regret", "cursor_ce", "mse"]:
            for family in protocol["families"]:
                pool = [r for r in searches if r["method"] == method and r["family"] == family]
                policies = ["realized", "rmse"] if method == "mse" else ["realized"]
                for policy in policies:
                    selected = (max(pool, key=lambda r: r["policies"][policy]["val_realized"]) if policy == "realized" else min(pool, key=lambda r: r["policies"][policy]["val_rmse"]))
                    point = selected["policies"][policy]
                    candidates.append({
                        "candidate": f"codex_{method}_{family}_{policy}", "method": method, "family": family, "selection_policy": policy,
                        "params": selected["params"], "search_job": selected["job_id"], "best_epoch": point["epoch"],
                        "final_epochs": scaled_count(point["epoch"]), "validation_realized": point["val_realized"], "validation_rmse": point["val_rmse"],
                        "search_initial_hash": selected["initial_state_sha256"],
                    })
        write_json(run_dir / "selection_frozen.json", {"frozen_utc": utcnow(), "candidates": candidates})
        primary = max([c for c in candidates if c["method"] == "soft_regret"], key=lambda c: c["validation_realized"])["candidate"]
        refit_cache = {}
        for candidate in candidates:
            key = json.dumps({k: candidate[k] for k in ["method", "family", "params", "final_epochs"]}, sort_keys=True)
            if key in refit_cache:
                result = refit_cache[key]
                Ledger(run_dir, protocol).add("refit_exact_cache", candidate=candidate["candidate"], shared_job=result["job_id"])
            else:
                result = execute_job(run_dir, protocol, {
                    "job_id": f"refit_{candidate['method']}_{candidate['family']}_{candidate['selection_policy']}",
                    "stage": "refit", "method": candidate["method"], "family": candidate["family"],
                    "params": candidate["params"], "epochs": candidate["final_epochs"],
                })
                refit_cache[key] = result
            candidate.update(checkpoint=result["checkpoint"], checkpoint_sha256=sha256(result["checkpoint"]), refit_initial_hash=result["initial_state_sha256"])
        guard = check_protected(run_dir)
        if not guard["unchanged"]:
            raise RuntimeError(f"reference inputs changed: {guard['changed']}")
        frozen = {"frozen_utc": utcnow(), "primary_candidate": primary, "candidates": candidates,
                  "protocol_sha256": sha256(run_dir / "protocol_frozen.json"), "holdout_evaluated": False}
        write_json(run_dir / "candidate_freeze.json", frozen)
        Ledger(run_dir, protocol).add("candidates_frozen", count=len(candidates), primary=primary, fingerprint=sha256(run_dir / "candidate_freeze.json"))
        snapshot(run_dir, protocol, "CANDIDATES_FROZEN; ready for one common 59-day evaluation")
        print("CANDIDATES FROZEN", primary, flush=True)
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "status", "check-protected"])
    args = parser.parse_args()
    protocol = read_json(PROTOCOL_PATH)
    run_dir = run_directory(protocol)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        print(json.dumps(prepare_data(run_dir), ensure_ascii=False, indent=2))
        snapshot(run_dir, protocol, "DATA_PREPARED; training not started")
    elif args.command == "train":
        train_all(run_dir, protocol)
    elif args.command == "check-protected":
        print(json.dumps(check_protected(run_dir), ensure_ascii=False))
    else:
        print(json.dumps(snapshot(run_dir, protocol, read_json(run_dir / "status.json")["stage"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

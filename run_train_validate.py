import argparse
import csv
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path

import portpicker
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn

from agent.learner import Learner
from environment import Env
from models import Model
import utils.soft_watkins as soft_watkins


# Paper values (Appendix A) that this repo can map directly.
PAPER_PARAMS = {
    "num_mixtures": 16,
    "bandit_beta": 1.0,
    "bandit_epsilon": 0.5,
    "max_discount": 0.9997,
    "min_discount": 0.97,
    "trace_length": 160,
    "burn_in": 0,
    "im_reward_scale_beta_im": 0.1,
    "priority_exponent": 0.6,
    "importance_sampling_exponent": 0.4,
    "batch_size": 64,
    "replay_capacity_trajectories": int(2e5),
    "rl_adam_lr": 3e-4,
    "lambda": 0.95,
    "kappa": 0.01,
    "tau": 0.25,
    "max_kl": 0.5,
}

# Legacy Gym Atari IDs used in older papers/code and their Gym 0.26 aliases.
ENV_ALIASES = {
    "AtlantisDeterministic-v4": "ALE/Atlantis-v5",
    "BreakoutDeterministic-v4": "ALE/Breakout-v5",
    "KrullDeterministic-v4": "ALE/Krull-v5",
}


def resolve_env_name(env_name: str) -> str:
    return ENV_ALIASES.get(env_name, env_name)


def apply_paper_hyperparams_to_repo() -> None:
    """Map paper hyperparameters onto knobs exposed by this repo."""
    Learner.beta = PAPER_PARAMS["im_reward_scale_beta_im"]
    Learner.discount_max = PAPER_PARAMS["max_discount"]
    Learner.discount_min = PAPER_PARAMS["min_discount"]
    Learner.bandit_beta = PAPER_PARAMS["bandit_beta"]
    Learner.bandit_epsilon = PAPER_PARAMS["bandit_epsilon"]
    Learner.lr = PAPER_PARAMS["rl_adam_lr"]

    # defaults order: lambda_, kappa, alpha, n, tau
    defaults = soft_watkins.compute_soft_watkins_loss.__defaults__
    soft_watkins.compute_soft_watkins_loss.__defaults__ = (
        PAPER_PARAMS["lambda"],
        PAPER_PARAMS["kappa"],
        defaults[2],
        defaults[3],
        PAPER_PARAMS["tau"],
    )


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def export_training_rows(train_log_path: Path, run_dir: Path) -> dict:
    """
    Export every raw training row into run artifacts:
    - training_rows_raw.log: exact copy of logger rows
    - training_rows.csv: parsed CSV with header for plotting
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_copy_path = run_dir / "training_rows_raw.log"
    csv_path = run_dir / "training_rows.csv"

    if train_log_path.exists():
        raw_text = train_log_path.read_text(encoding="utf-8")
    else:
        raw_text = ""

    raw_copy_path.write_text(raw_text, encoding="utf-8")

    header = [
        "elapsed_time",
        "total_updates",
        "total_frames",
        "loss",
        "policy_loss",
        "intrinsic_loss",
        "reward",
        "intrinsic",
        "epsilon",
        "arm",
        "replay_ratio",
    ]
    row_count = 0
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != len(header):
                continue
            writer.writerow(parts)
            row_count += 1

    return {
        "training_raw_log_copy": str(raw_copy_path),
        "training_rows_csv": str(csv_path),
        "training_row_count": row_count,
    }


def run_training_until_frames(
    learner: Learner,
    max_frames: int,
    checkpoint_path: str,
    run_dir: str,
) -> None:
    learner.replay_buffer.start_threads()

    # Mirror Learner.run() setup.
    request_thread = threading.Thread(target=learner.answer_requests, daemon=True)
    request_thread.start()

    prepare_thread = threading.Thread(target=learner.prepare_data, daemon=True)
    prepare_thread.start()

    while learner.replay_buffer.logger.total_frames < max_frames:
        while not learner.batch_data:
            time.sleep(0.001)
        block = learner.batch_data.pop(0)
        learner.update(block)

    # Stop accepting new actor RPC work before checkpoint/shutdown.
    learner.stopping = True

    Learner.save(learner.model, checkpoint_path)
    logger = learner.replay_buffer.logger
    train_log_path = f"logs/{logger.datetime}"
    run_path = Path(run_dir)
    exported_logs = export_training_rows(Path(train_log_path), run_path)
    summary = {
        "target_frames": max_frames,
        "final_frames": int(logger.total_frames),
        "final_updates": int(logger.total_updates),
        "final_loss": float(logger.loss),
        "final_policy_loss": float(logger.p_loss),
        "final_intrinsic_loss": float(logger.intr_loss),
        "final_reward": float(logger.reward),
        "final_intrinsic": float(logger.intrinsic),
        "final_epsilon": float(logger.epsilon),
        "final_arm": float(logger.arm),
        "final_replay_ratio": float(logger.replay_ratio),
        "checkpoint_path": checkpoint_path,
        "training_log_path": train_log_path,
        **exported_logs,
    }
    with Path(run_path, "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_jsonl(
        Path(run_path, "events.jsonl"),
        {
            "event": "training_complete",
            "timestamp": datetime.now().isoformat(),
            **summary,
        },
    )
    print(
        f"Reached target frames={logger.total_frames}. "
        f"Checkpoint saved to '{checkpoint_path}'.",
        flush=True,
    )


def run_worker(
    rank: int,
    env_name: str,
    num_actors: int,
    buffer_size: int,
    batch_size: int,
    burnin: int,
    rollout: int,
    max_frames: int,
    checkpoint_path: str,
    run_dir: str,
    mode: str,
) -> None:
    world_size = 1 + num_actors
    node_name = "learner" if rank == 0 else f"actor{rank - 1}"
    rpc.init_rpc(node_name, rank=rank, world_size=world_size)

    if rank == 0:
        learner = Learner(
            env_name=env_name,
            N=num_actors,
            size=buffer_size,
            B=batch_size,
            burnin=burnin,
            rollout=rollout,
            mode=mode,
        )
        run_training_until_frames(
            learner=learner,
            max_frames=max_frames,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
        )

    rpc.shutdown()


@torch.no_grad()
def evaluate_checkpoint(
    env_name: str,
    checkpoint_path: str,
    num_actors: int,
    episodes: int,
    max_steps_per_episode: int,
    device: str,
    run_dir: str,
) -> dict:
    env = Env(env_name, render_mode=None)
    model = nn.DataParallel(Model(N=num_actors, action_size=env.action_size)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(run_dir, "validation_episodes.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("episode,return,steps\n")

    returns = []
    for ep in range(episodes):
        obs = env.reset()
        state = (
            torch.zeros((1, 512), device=device),
            torch.zeros((1, 512), device=device),
        )

        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            _, pi, state = model(obs_t, state)
            action = torch.argmax(pi[0, 0, :]).item()  # arm 0 as exploit policy

            obs, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        returns.append(total_reward)
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(f"{ep + 1},{total_reward:.6f},{steps}\n")
        append_jsonl(
            Path(run_dir, "events.jsonl"),
            {
                "event": "validation_episode",
                "timestamp": datetime.now().isoformat(),
                "episode": ep + 1,
                "return": float(total_reward),
                "steps": int(steps),
            },
        )
        print(f"[validation] episode={ep + 1} return={total_reward:.2f} steps={steps}", flush=True)

    mean_return = sum(returns) / len(returns) if returns else 0.0
    summary = {
        "env_name": env_name,
        "episodes": episodes,
        "mean_return": float(mean_return),
        "min_return": float(min(returns)) if returns else 0.0,
        "max_return": float(max(returns)) if returns else 0.0,
        "checkpoint_path": checkpoint_path,
        "episodes_csv": str(csv_path),
    }
    with Path(run_dir, "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    append_jsonl(
        Path(run_dir, "events.jsonl"),
        {
            "event": "validation_complete",
            "timestamp": datetime.now().isoformat(),
            **summary,
        },
    )
    print(
        f"[validation] env={env_name} episodes={episodes} mean_return={mean_return:.2f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train until frame budget, then run validation.")
    parser.add_argument("--env", default="BreakoutDeterministic-v4")
    parser.add_argument("--max-frames", type=int, default=2_000_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=20_000)
    parser.add_argument("--checkpoint-path", default="saved/final_2m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", default="")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "no_trust_region", "no_soft_watkins"],
    )
    args = parser.parse_args()

    apply_paper_hyperparams_to_repo()
    resolved_env = resolve_env_name(args.env)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or str(Path("results") / f"train_validate_{run_id}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with Path(run_dir, "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "env_input": args.env,
                "env_resolved": resolved_env,
                "max_frames": args.max_frames,
                "eval_episodes": args.eval_episodes,
                "eval_max_steps": args.eval_max_steps,
                "checkpoint_path": args.checkpoint_path,
                "device": args.device,
                "mode": args.mode,
                "paper_params": PAPER_PARAMS,
            },
            f,
            indent=2,
        )

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(portpicker.pick_unused_port())

    mp.spawn(
        run_worker,
        args=(
            resolved_env,
            PAPER_PARAMS["num_mixtures"],
            PAPER_PARAMS["replay_capacity_trajectories"],
            PAPER_PARAMS["batch_size"],
            PAPER_PARAMS["burn_in"],
            PAPER_PARAMS["trace_length"],
            args.max_frames,
            args.checkpoint_path,
            run_dir,
            args.mode,
        ),
        nprocs=1 + PAPER_PARAMS["num_mixtures"],
        join=True,
    )

    summary = evaluate_checkpoint(
        env_name=resolved_env,
        checkpoint_path=args.checkpoint_path,
        num_actors=PAPER_PARAMS["num_mixtures"],
        episodes=args.eval_episodes,
        max_steps_per_episode=args.eval_max_steps,
        device=args.device,
        run_dir=run_dir,
    )
    print(f"[artifacts] saved to {run_dir}", flush=True)
    print(
        "[artifacts] files: config.json, training_summary.json, "
        "validation_episodes.csv, validation_summary.json, events.jsonl",
        flush=True,
    )


if __name__ == "__main__":
    main()

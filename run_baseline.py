import importlib
import argparse

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

TARGET_ENVS = list(ENV_ALIASES.keys()) + list(ENV_ALIASES.values())

PAPER_PARAMS

from agent.learner import Learner
import utils.soft_watkins as soft_watkins


def apply_paper_hyperparams_to_repo() -> None:
    """Map paper hyperparameters onto knobs exposed by this repo."""
    Learner.beta = PAPER_PARAMS["im_reward_scale_beta_im"]
    Learner.discount_max = PAPER_PARAMS["max_discount"]
    Learner.discount_min = PAPER_PARAMS["min_discount"]
    Learner.bandit_beta = PAPER_PARAMS["bandit_beta"]
    Learner.bandit_epsilon = PAPER_PARAMS["bandit_epsilon"]
    Learner.lr = PAPER_PARAMS["rl_adam_lr"]

    # The implementation already defaults to these values; set explicitly for clarity.
    defaults = soft_watkins.compute_soft_watkins_loss.__defaults__
    # defaults order: lambda_, kappa, alpha, n, tau
    soft_watkins.compute_soft_watkins_loss.__defaults__ = (
        PAPER_PARAMS["lambda"],
        PAPER_PARAMS["kappa"],
        defaults[2],
        defaults[3],
        PAPER_PARAMS["tau"],
    )


def load_entrypoint():
    # Import the real module so torch.multiprocessing can pickle worker targets.
    return importlib.import_module("meme_entry")


def run_training(env_name: str, mode: str = "baseline"):
    """
    Launches the repo's distributed training loop.
    Note: training is open-ended; stop manually when desired.
    """
    import gym

    if env_name not in TARGET_ENVS:
        raise ValueError(f"Use one of: {TARGET_ENVS}")

    resolved_env_name = ENV_ALIASES.get(env_name, env_name)
    if resolved_env_name not in gym.envs.registry:
        raise ValueError(
            f"Environment '{resolved_env_name}' is not registered in this kernel. "
            "Install gym Atari extras and ROMs, then restart the kernel."
        )

    entry = load_entrypoint()
    entry.main(
        env_name=resolved_env_name,
        num_actors=PAPER_PARAMS["num_mixtures"],
        buffer_size=PAPER_PARAMS["replay_capacity_trajectories"],
        batch_size=PAPER_PARAMS["batch_size"],
        burnin=PAPER_PARAMS["burn_in"],
        rollout=PAPER_PARAMS["trace_length"],
        mode=mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MEME baseline/ablations.")
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "no_trust_region", "no_soft_watkins"],
        help="Training mode variant.",
    )
    parser.add_argument(
        "--env",
        default="BreakoutDeterministic-v4",
        choices=TARGET_ENVS,
        help="Environment ID (legacy or ALE form).",
    )
    args = parser.parse_args()

    apply_paper_hyperparams_to_repo()
    print("Paper hyperparameters applied to Learner and Soft-Watkins defaults.")
    print(f"Running mode: {args.mode}")

    # Run one environment at a time (each call blocks until you stop execution):
    run_training(args.env, mode=args.mode)


"""PPO training for RaceEnv (f1tenth_gym) — CleanRL-style, gymnasium 1.3.0.

Design notes (read before touching):
  * Vec env = AsyncVectorEnv(context="fork"). Forked workers inherit the parent's
    compiled numba njit cache via copy-on-write, so we warm it up in the parent
    FIRST (_warmup_numba) to avoid an N-worker recompilation stall at startup.
  * Normalization is applied as VECTOR wrappers (single obs_rms in the main proc),
    NOT per-env. Per-env NormalizeObservation under multiprocessing keeps divergent
    stats per worker and is not exportable. The obs_rms is persisted in the
    checkpoint and folded into the ONNX export (ExportActor) so inference feeds
    RAW observations.
  * CUDA is initialised AFTER the workers are forked. Calling torch.cuda.* before a
    fork breaks the child CUDA context (deadlock).

Usage:
    docker compose run --rm train
    docker compose run --rm sim python test_RL/export_onnx.py

Version assumption: gymnasium >= 1.0. The vector-wrapper module path
(gymnasium.wrappers.vector.*) and gym.vector.AutoresetMode are 1.x API.
  * AutoresetMode: NEXT_STEP (required by vector NormalizeObservation). With NEXT_STEP,
    next_obs already contains the reset obs of the new episode when done=True, so the
    CleanRL rollout loop needs no modification — the obs at t+1 is always valid.
"""

import os

# IMPORTANT: thread limits must be set BEFORE importing numpy/torch — OpenMP/MKL
# read them at import time. Setting them later is a no-op. This caps BLAS thread
# pools so the per-env numba workers and the main-proc torch update don't each
# fan out across all cores (oversubscription pathology on many-core CPUs).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Band-aid for legacy f1tenth_gym deps that still reference np.float/np.int.
# TODO(B12): remove once legacy-gym path is purged.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=DeprecationWarning)

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
MODEL_DIR = current_dir / "models"
TB_LOG_DIR = current_dir / "tb_logs"


# ── Args ────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = "racecar_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    vec: str = "async"          # "async" (fork, parallel) | "sync" (single proc, debug)
    warmup: bool = True         # precompile njit in parent before fork
    torch_threads: int = 4      # intra-op torch threads on CPU; tiny MLP -> keep LOW (try 1/2/4)

    total_timesteps: int = 1_000_000_000
    num_envs: int = 8
    num_steps: int = 2048
    anneal_lr: bool = True
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001     # std is learned; entropy bonus inflates sigma in continuous
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    encoder_layer_sizes: str = "128,128"
    logstd_init: float = -0.5   # sigma ~ 0.61, avoids saturating ClipAction on [-1,1]

    # runtime-filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    save_freq: int = 500_000


# ── Env construction ─────────────────────────────────────────────────────────

def _ensure_root_on_path():
    import sys
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def make_env(seed: int, idx: int):
    def thunk():
        _ensure_root_on_path()
        from test_RL.env import RaceEnv
        env = RaceEnv()
        env = gym.wrappers.ClipAction(env)      # stateless, safe per-env
        env.action_space.seed(seed + idx)
        return env
    return thunk


def _warmup_numba():
    """Trigger f1tenth njit compilation in the parent; forked children then
    inherit the compiled cache via COW instead of recompiling N times."""
    _ensure_root_on_path()
    from test_RL.env import RaceEnv
    e = RaceEnv()
    e.reset(seed=0)
    e.step(e.action_space.sample())
    e.close()


def build_envs(args: Args):
    fns = [make_env(args.seed, i) for i in range(args.num_envs)]
    next_step = gym.vector.AutoresetMode.NEXT_STEP
    if args.vec == "sync" or args.num_envs == 1:
        envs = gym.vector.SyncVectorEnv(fns, autoreset_mode=next_step)
    else:
        envs = gym.vector.AsyncVectorEnv(fns, context="fork", autoreset_mode=next_step)

    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.vector.NormalizeObservation(envs)
    obs_norm = envs                                   # handle for obs_rms export
    envs = gym.wrappers.vector.NormalizeReward(envs, gamma=args.gamma)
    return envs, obs_norm


# ── Networks ──────────────────────────────────────────────────────────────────

def layer_init(layer: nn.Module, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,
                 layer_sizes: Tuple[int, ...] = (128, 128), logstd_init: float = -0.5):
        super().__init__()
        layers, last = [], obs_dim
        for h in layer_sizes:
            layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
            last = h
        self.encoder = nn.Sequential(*layers)
        self.actor_mean = layer_init(nn.Linear(last, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * logstd_init)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(last, 64), std=1.0), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        # clamp matches the ONNX export graph (consistency train<->inference)
        return self.encoder(torch.clamp(x, -10.0, 10.0))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._feat(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        f = self._feat(x)
        mean = self.actor_mean(f)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(f)


class ExportActor(nn.Module):
    """Deterministic policy with obs normalization embedded for ONNX export.

    Inference graph: raw_obs -> (obs-mean)/sqrt(var+eps) -> clamp(+-10) -> MLP -> action_mean.
    export_onnx.py should build this from a checkpoint (state_dict + obs_rms) and
    torch.onnx.export it. Students then feed RAW observations at inference.
    """
    def __init__(self, agent: Agent, obs_mean, obs_var, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(obs_var, dtype=torch.float32))
        self.eps = float(eps)
        self.encoder = agent.encoder
        self.actor_mean = agent.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / torch.sqrt(self.var + self.eps)
        x = torch.clamp(x, -10.0, 10.0)
        return self.actor_mean(self.encoder(x))


def export_actor(agent: Agent, obs_rms) -> ExportActor:
    """Factory for export_onnx.py. obs_rms is the RunningMeanStd from the
    NormalizeObservation vector wrapper (has .mean / .var)."""
    return ExportActor(agent, obs_rms.mean, obs_rms.var).eval()


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(str(TB_LOG_DIR / run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join(f"|{k}|{v}|" for k, v in vars(args).items())),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ── Build envs BEFORE any CUDA call (fork-safety) ──────────────────────────
    if args.warmup and args.vec != "sync":
        print("[warmup] precompiling njit in parent before fork ...")
        t0 = time.time()
        _warmup_numba()
        print(f"[warmup] done in {time.time() - t0:.1f}s")

    envs, obs_norm = build_envs(args)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous action space only"
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    # ── Now it is safe to init CUDA ────────────────────────────────────────────
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    # On CPU, cap torch intra-op threads. The MLP is tiny (102->128->128); fanning
    # out over all cores makes thread launch dominate compute (10-50x slowdown on
    # minibatch matmuls). set_num_threads at runtime is reliable; env vars are not.
    if device.type == "cpu":
        torch.set_num_threads(args.torch_threads)
    print(f"[threads] torch intra-op = {torch.get_num_threads()}")

    layer_sizes = tuple(int(s) for s in args.encoder_layer_sizes.split(","))
    agent = Agent(obs_dim, action_dim, layer_sizes, args.logstd_init).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ── Rollout storage ────────────────────────────────────────────────────────
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    last_save = 0
    episodic_returns = []  # accumulate episode returns per iteration for mean logging

    def save_checkpoint(tag: str):
        ckpt = {
            "state_dict": agent.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "global_step": global_step,
            # critical for ONNX export: obs normalization stats
            "obs_rms_mean": np.asarray(obs_norm.obs_rms.mean, dtype=np.float32),
            "obs_rms_var": np.asarray(obs_norm.obs_rms.var, dtype=np.float32),
        }
        torch.save(ckpt, str(MODEL_DIR / f"racecar_ppo_{tag}.pt"))

    try:
        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            _t_roll = time.perf_counter()
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs_np, reward_np, term, trunc, infos = envs.step(action.cpu().numpy())
                reward_np = np.clip(reward_np, -10.0, 10.0)          # post-NormalizeReward safety clip
                done_np = np.logical_or(term, trunc)

                rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
                next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

                # Episode stats: gymnasium 1.x vector RecordEpisodeStatistics ->
                # infos["episode"] arrays + infos["_episode"] mask. (No more final_info.)
                if "episode" in infos:
                    mask = infos.get("_episode", np.ones(args.num_envs, dtype=bool))
                    for i in np.nonzero(mask)[0]:
                        ep_r = float(infos["episode"]["r"][i])
                        ep_l = int(infos["episode"]["l"][i])
                        print(f"  global_step={global_step}, episodic_return={ep_r:.2f}, length={ep_l}")
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)
                        episodic_returns.append(ep_r)

            # ── GAE ────────────────────────────────────────────────────────────
            _t_roll = time.perf_counter() - _t_roll
            _t_upd = time.perf_counter()
            # NOTE: term and trunc are both masked as terminal (classic CleanRL).
            # If RaceEnv truncates on a step/time limit and you want correctness,
            # bootstrap V(final_obs) on truncation instead of zeroing it.
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0.0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            b_obs = obs.reshape(-1, obs_dim)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, action_dim)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(args.batch_size)
            clipfracs = []
            approx_kl = old_approx_kl = torch.tensor(0.0)
            for _ in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb = b_inds[start:start + args.minibatch_size]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_adv = b_advantages[mb]
                    if args.norm_adv:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_unclipped = (newvalue - b_returns[mb]) ** 2
                        v_clipped = b_values[mb] + torch.clamp(
                            newvalue - b_values[mb], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                        v_loss = 0.5 * torch.max(v_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            _t_upd = time.perf_counter() - _t_upd

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Per-iteration mean episodic return across all completed episodes
            if len(episodic_returns) > 0:
                writer.add_scalar("charts/episodic_return_mean", np.mean(episodic_returns), global_step)
                print(f"  [iteration] meanepisodic_return={np.mean(episodic_returns):.2f} (n={len(episodic_returns)})")
                episodic_returns = []

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("perf/rollout_s", _t_roll, global_step)
            writer.add_scalar("perf/update_s", _t_upd, global_step)
            print(f"Iteration {iteration}/{args.num_iterations}, SPS: {sps} "
                  f"| rollout={_t_roll:.1f}s update={_t_upd:.1f}s")
            print("-----------------------")

            if global_step - last_save >= args.save_freq:
                save_checkpoint(str(global_step))
                save_checkpoint("latest")
                last_save = global_step
                print(f"[Checkpoint] {global_step} timesteps saved.")

    except KeyboardInterrupt:
        print("\nInterrupted — saving fallback ...")
    finally:
        try:
            save_checkpoint("final")
        except Exception as e:
            print(f"[shutdown] checkpoint save failed: {e}")
        try:
            envs.close(terminate=True)
        except Exception as e:
            print(f"[shutdown] envs.close failed: {e}")
        try:
            writer.close()
        except Exception:
            pass
        print("Training complete. Final checkpoint saved (incl. obs_rms for ONNX).")


if __name__ == "__main__":
    main()
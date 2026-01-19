import torch
import torch.nn as nn
import math
import numpy as np
import argparse
from types import SimpleNamespace
import time

# ============ LOAD CHECKPOINT ============
def dict_to_namespace(d):
    """Convert dictionary to SimpleNamespace for easy attribute access"""
    return SimpleNamespace(**d)

# ============ RECREATE MODELS ============
def recreate_models_from_checkpoint(checkpoint, env, device, agent_type="fasttd3"):
    """Reconstruct actor from checkpoint"""
    
    args_dict = checkpoint["args"]
    args = dict_to_namespace(args_dict) if isinstance(args_dict, dict) else args_dict
    
    n_act = env.num_actions
    n_obs = env.num_obs if type(env.num_obs) == int else env.num_obs[0]
    
    if env.asymmetric_obs:
        n_critic_obs = (
            env.num_privileged_obs
            if type(env.num_privileged_obs) == int
            else env.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    
    print(f"n_obs: {n_obs}, n_act: {n_act}")
    print(f"Training num_envs: {args.num_envs}")
    
    # Recreate actor
    if agent_type == "fasttd3":
        from fast_td3 import Actor, MultiTaskActor
        
        actor_kwargs = {
            "n_obs": n_obs,
            "n_act": n_act,
            "num_envs": args.num_envs,
            "device": device,
            "init_scale": args.init_scale,
            "hidden_dim": args.actor_hidden_dim,
            "std_min": args.std_min,
            "std_max": args.std_max,
        }
        
        if hasattr(env, 'num_tasks'):
            actor_kwargs["n_obs"] = n_obs - env.num_tasks + args.task_embedding_dim
            actor_kwargs["num_tasks"] = env.num_tasks
            actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
            actor = MultiTaskActor(**actor_kwargs)
        else:
            actor = Actor(**actor_kwargs)
    
    elif agent_type == "fasttd3_simbav2":
        from fast_td3_simbav2 import Actor, MultiTaskActor
        
        actor_kwargs = {
            "n_obs": n_obs,
            "n_act": n_act,
            "num_envs": args.num_envs,
            "device": device,
            "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
            "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
            "alpha_init": 1.0 / (args.actor_num_blocks + 1),
            "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
            "expansion": 4,
            "c_shift": 3.0,
            "num_blocks": args.actor_num_blocks,
            "hidden_dim": args.actor_hidden_dim,
        }
        
        if hasattr(env, 'num_tasks'):
            actor_kwargs["n_obs"] = n_obs - env.num_tasks + args.task_embedding_dim
            actor_kwargs["num_tasks"] = env.num_tasks
            actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
            actor = MultiTaskActor(**actor_kwargs)
        else:
            actor = Actor(**actor_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    
    print(f"✓ Actor loaded")
    
    from fast_td3_utils import EmpiricalNormalization
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    obs_normalizer.eval()
    
    return actor, obs_normalizer, args

# ============ RUN POLICY ON ISAAC SIM ============
def run_policy_on_isaac_sim(model_path, env_name, num_episodes=5, deterministic=True, headless=False):
    """
    Run the policy on Isaac Sim with visualization
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Headless mode: {headless}")
    
    print(f"\nLoading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args_dict = checkpoint["args"]
    args = dict_to_namespace(args_dict) if isinstance(args_dict, dict) else args_dict
    
    print(f"Environment: {env_name}")
    print(f"Agent: {args.agent}")
    
    # Create Isaac Lab environment with visualization
    print(f"\nInitializing Isaac Lab environment...")
    try:
        from environments.isaaclab_env import IsaacLabEnv
        
        action_bounds = getattr(args, "action_bounds", [-1.0, 1.0])
        
        # Create environment
        env = IsaacLabEnv(
            env_name,
            device.type,
            num_envs=1,
            seed=42,
            action_bounds=action_bounds,
        )
        
        print(f"✓ Isaac Lab environment initialized")
        print(f"  - Max episode steps: {env.max_episode_steps}")
        print(f"  - Observation shape: {env.num_obs}")
        print(f"  - Action shape: {env.num_actions}")
        
    except Exception as e:
        print(f"✗ Error creating Isaac Lab environment:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load policy
    print(f"\nLoading policy...")
    try:
        actor, obs_normalizer, args = recreate_models_from_checkpoint(
            checkpoint, env, device, agent_type=args.agent
        )
        print(f"✓ Policy loaded successfully")
    except Exception as e:
        print(f"✗ Error loading policy:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✓ Running {num_episodes} episodes on Isaac Sim (deterministic={deterministic})")
    print("="*70)
    
    episode_returns = []
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print("-" * 70)
            
            # Reset environment
            try:
                print(f"  Resetting environment...")
                obs = env.reset(random_start_init=False)
                print(f"  ✓ Reset complete, obs shape: {obs.shape}")
            except Exception as e:
                print(f"  ✗ Error resetting environment: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            episode_return = 0.0
            episode_length = 0
            dones = None
            step_errors = 0
            
            while episode_length < env.max_episode_steps:
                try:
                    # Get action
                    with torch.no_grad():
                        norm_obs = obs_normalizer(obs, update=False)
                        
                        if deterministic:
                            actions = actor(norm_obs)
                        else:
                            if dones is None:
                                dones = torch.zeros(1, device=obs.device, dtype=torch.bool)
                            actions = actor.explore(norm_obs, dones=dones, deterministic=False)
                    
                    # Step environment
                    next_obs, rewards, dones, infos = env.step(actions.float())
                    
                    episode_return += rewards.item() if rewards.dim() > 0 else rewards
                    episode_length += 1
                    
                    # Print progress every 50 steps
                    if episode_length % 50 == 0:
                        print(f"    Step {episode_length}/{env.max_episode_steps}, Reward: {episode_return:.2f}")
                    
                    obs = next_obs
                    
                    if dones.item():
                        print(f"    Episode terminated by environment")
                        break
                
                except Exception as e:
                    step_errors += 1
                    if step_errors > 5:
                        print(f"  ✗ Too many step errors, skipping episode")
                        break
                    print(f"  ✗ Error in step {episode_length}: {type(e).__name__}: {e}")
                    time.sleep(0.1)
                    continue
            
            episode_returns.append(episode_return)
            print(f"  ✓ Episode finished: {episode_length} steps, Total Return: {episode_return:.2f}")
    
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during episodes: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Print summary
        if episode_returns:
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"Episodes completed: {len(episode_returns)}")
            print(f"Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
            print(f"Min Return:     {np.min(episode_returns):.2f}")
            print(f"Max Return:     {np.max(episode_returns):.2f}")
            print("="*70)
        
        try:
            env.close()
        except:
            pass

# ============ COMMAND LINE ARGS ============
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--env", type=str, required=True, help="Isaac Lab environment name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    return parser.parse_args()

# ============ MAIN ============
if __name__ == "__main__":
    args = get_args()
    
    run_policy_on_isaac_sim(
        model_path=args.model_path,
        env_name=args.env,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        headless=args.headless,
    )

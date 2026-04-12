import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from env.rl_wrapper import TriageRLWrapper

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "triagerl_dqn")

def main():
    print("Initializing TriageRL Environment for training...")
    # Wrap in a vectorized env, required for standard SB3 training
    # We pass training_mode=True to mock LLM actions automatically
    def make_env():
        return TriageRLWrapper(training_mode=True)
        
    env = make_vec_env(make_env, n_envs=4)

    print("Initializing DQN Agent...")
    # Using DQN because our action space is discrete (80 unique actions)
    # We use a Multi-Layer Perceptron (MLP) policy for the 22-dimensional state vector
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
    )

    print("Starting Training (this may take a few minutes)...")
    # 50,000 steps is sufficient for a simple discrete environment to converge nicely
    model.learn(total_timesteps=50_000, progress_bar=True)

    print(f"Saving model to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    
    print("Evaluating trained model...")
    eval_env = TriageRLWrapper(training_mode=True)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    print(f"Evaluation complete. Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()

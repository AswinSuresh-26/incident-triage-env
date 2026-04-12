import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.environment import IncidentEnv
from env.models import Action
from env.tasks import VALID_ACTIONS

# Sorted for deterministic mapping
TARGETS = sorted([
    "api_gateway", "auth_service", "database",
    "web_server", "worker_pool", "cache",
    "payment_service", "order_service", "inventory_service",
    "ops_team"
])

DIFFICULTIES = ["easy", "medium", "hard"]

class TriageRLWrapper(gym.Env):
    """
    Gymnasium wrapper around IncidentEnv to train the TriageRL adaptive policy.
    The agent observes the environment state and the LLM's suggested action,
    then executes an action (which may override the LLM's suggestion).
    """
    
    def __init__(self, training_mode: bool = False):
        super().__init__()
        self.incident_env = IncidentEnv()
        self.llm_action = None
        self.training_mode = training_mode
        self.rng = np.random.default_rng()
        
        # Dimensions: 
        # difficulty (3) + step (1) + llm_action (8) + llm_target (10) = 22
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(22,), dtype=np.float32
        )
        
        # Action space: combination of action_type x target
        self.action_space = spaces.Discrete(len(VALID_ACTIONS) * len(TARGETS))
    
    def seed(self, seed=None):
        self.incident_env = IncidentEnv(seed=seed)
        self.rng = np.random.default_rng(seed)
        
    def _generate_mock_llm_action(self):
        """Simulate an LLM suggestion. 70% chance of optimal, 30% random."""
        try:
            task = self.incident_env.get_state()
            # It's an internal protected method, but we can peek the solution for training mock
            solution = self.incident_env._current_task["solution"]
            
            if self.rng.random() < 0.70:
                self.llm_action = Action(action_type=solution["action_type"], target=solution["target"])
            else:
                self.llm_action = Action(
                    action_type=self.rng.choice(VALID_ACTIONS),
                    target=self.rng.choice(TARGETS)
                )
        except Exception:
            self.llm_action = Action(action_type="inspect_logs", target="api_gateway")
        
    def _action_to_idx(self, action_type: str, target: str) -> int:
        if action_type not in VALID_ACTIONS or target not in TARGETS:
            return 0  # default fallback if unknown
        a_idx = VALID_ACTIONS.index(action_type)
        t_idx = TARGETS.index(target)
        return a_idx * len(TARGETS) + t_idx
        
    def _idx_to_action(self, idx: int) -> Action:
        a_idx = idx // len(TARGETS)
        t_idx = idx % len(TARGETS)
        return Action(action_type=VALID_ACTIONS[a_idx], target=TARGETS[t_idx])

    @classmethod
    def build_obs_vector(cls, task_id: str, step_num: int, llm_action: Action | None) -> np.ndarray:
        """Create the 22-dimensional observation vector for SB3 inference."""
        obs_vec = np.zeros(22, dtype=np.float32)
        
        # 1. Difficulty (One-Hot) - cols 0-2
        if task_id in DIFFICULTIES:
            obs_vec[DIFFICULTIES.index(task_id)] = 1.0
            
        # 2. Step Ratio - col 3
        obs_vec[3] = min(1.0, step_num / 10.0)
        
        # 3. LLM Suggested Action - cols 4-11
        if llm_action:
            if llm_action.action_type in VALID_ACTIONS:
                obs_vec[4 + VALID_ACTIONS.index(llm_action.action_type)] = 1.0
                
        # 4. LLM Suggested Target - cols 12-21
        if llm_action:
            if llm_action.target in TARGETS:
                obs_vec[12 + TARGETS.index(llm_action.target)] = 1.0
                
        return obs_vec

    def _get_obs(self) -> np.ndarray:
        state = self.incident_env.get_state()
        return self.build_obs_vector(
            task_id=state.get("task_id", ""),
            step_num=state.get("step", 0),
            llm_action=self.llm_action
        )

    def set_llm_action(self, action: Action):
        """Called before env.step() so the agent can react to the LLM's plan."""
        self.llm_action = action

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            
        self.incident_env.reset()
        if self.training_mode:
            self._generate_mock_llm_action()
        else:
            self.llm_action = None
            
        return self._get_obs(), {}

    def step(self, action_idx: int):
        action = self._idx_to_action(action_idx)
        try:
            _, reward, done, info = self.incident_env.step(action)
        except Exception:
            # Penalize invalid execution
            return self._get_obs(), 0.01, True, False, {}
            
        if not done and self.training_mode:
            self._generate_mock_llm_action()
            
        return self._get_obs(), float(reward), done, False, info

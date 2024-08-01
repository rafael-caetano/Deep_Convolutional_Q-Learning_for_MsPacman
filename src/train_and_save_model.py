import gymnasium as gym
from .DCQN_Agent import Agent
from .DCQN_Training import DCQN_Train
import pickle
import os

def train_and_save_model(use_dueling, model_name, run_number):
    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
    number_actions = env.action_space.n
    
    agent = Agent(number_actions, learning_rate=5e-4, minibatch_size=64, discount_factor=0.99, use_dueling=use_dueling)
    trainer = DCQN_Train(env, agent, number_episodes=2000, max_timesteps_per_episode=10000,
                         epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    scores = trainer.train()

    # Save model to outputs folder
    model_filename = os.path.join('outputs', f'{model_name}_run_{run_number}.pth')
    trainer.save_model(model_filename)
    print(f"Model saved to {model_filename}")

    # Save scores to a pickle file in outputs folder
    scores_filename = os.path.join('outputs', f'{model_name}_scores_run_{run_number}.pkl')
    with open(scores_filename, 'wb') as f:
        pickle.dump(scores, f)
    print(f"Scores saved to {scores_filename}")
    
    return scores
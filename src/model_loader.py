import torch
import gymnasium as gym
from .DCQN_Network import Network
from .DCQN_Dueling_Network import DuelingDQN

def load_model(filename, action_size, use_dueling=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if use_dueling:
        model = DuelingDQN(action_size).to(device)
    else:
        model = Network(action_size).to(device)
    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def use_model(model, env_name, num_episodes=5):
    env = gym.make(env_name, render_mode='rgb_array')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
            with torch.no_grad():
                action_values = model(state_tensor)
            action = action_values.max(1)[1].item()
            
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

def load_and_use_models(original_model_path, dueling_model_path, env_name, num_episodes=5):
    action_size = gym.make(env_name).action_space.n

    # Load and use original DQN model
    original_model = load_model(original_model_path, action_size, use_dueling=False)
    print("Using Original DQN Model:")
    use_model(original_model, env_name, num_episodes)

    # Load and use Dueling DQN model
    dueling_model = load_model(dueling_model_path, action_size, use_dueling=True)
    print("\nUsing Dueling DQN Model:")
    use_model(dueling_model, env_name, num_episodes)
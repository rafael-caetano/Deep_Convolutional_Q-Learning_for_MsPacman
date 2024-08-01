import gymnasium as gym
import torch
import numpy as np
import imageio
import io
import base64
from IPython.display import HTML, display
from src.DCQN_Network import Network
from src.DCQN_Dueling_Network import DuelingDQN
from src.Preprocess_Frame_Inputs import preprocess_frame

def create_best_video(env_name, model=None, video_filename='video.mp4', fps=30, max_steps=10000, num_episodes=5, epsilon=0.05):
    env = gym.make(env_name, render_mode='rgb_array', full_action_space=False)
    best_score = float('-inf')
    best_frames = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        frames = []
        step = 0

        while not done and step < max_steps:
            frame = env.render()
            frames.append(frame)
            
            if model is None or (np.random.random() < epsilon):
                # Random action
                action = env.action_space.sample()
            else:
                # Use the model to choose an action
                state_tensor = preprocess_frame(state).to(model.device)
                with torch.no_grad():
                    action_values = model(state_tensor)
                action = action_values.max(1)[1].item()
            
            state, reward, done, _, _ = env.step(action)
            score += reward
            step += 1

        print(f"Episode {episode + 1}/{num_episodes} - Score: {score}, Steps: {step}")

        if score > best_score:
            best_score = score
            best_frames = frames

    env.close()
    imageio.mimsave(video_filename, best_frames, fps=fps)
    print(f"Best video saved as {video_filename} (Best score: {best_score})")
    return best_score

def display_video(video_filename='video.mp4'):
    video = io.open(video_filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''
        <video alt="test" controls style="height: 400px;">
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        </video>'''.format(encoded.decode('ascii')))

def load_model(filename, action_size, use_dueling=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if use_dueling:
        model = DuelingDQN(action_size).to(device)
    else:
        model = Network(action_size).to(device)
    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {filename}")
    print(f"Model structure: {model}")
    
    return model

def create_and_display_videos(env_name, num_episodes=5):
    env = gym.make(env_name, render_mode='rgb_array', full_action_space=False)
    action_size = env.action_space.n
    print(f"Action space size: {action_size}")
    env.close()

    # Create video for random agent
    print("Creating video for Random Agent...")
    random_score = create_best_video(env_name, model=None, video_filename='outputs/random_agent_video.mp4', num_episodes=num_episodes)
    display(display_video('outputs/random_agent_video.mp4'))

    # Load and create video for Original DQN
    print("\nCreating video for Original DQN...")
    original_model = load_model('outputs/original_dqn_run_1.pth', action_size, use_dueling=False)
    original_score = create_best_video(env_name, model=original_model, video_filename='outputs/original_dqn_video.mp4', num_episodes=num_episodes)
    display(display_video('outputs/original_dqn_video.mp4'))

    # Load and create video for Dueling DQN
    print("\nCreating video for Dueling DQN...")
    dueling_model = load_model('outputs/dueling_dqn_run_1.pth', action_size, use_dueling=True)
    dueling_score = create_best_video(env_name, model=dueling_model, video_filename='outputs/dueling_dqn_video.mp4', num_episodes=num_episodes)
    display(display_video('outputs/dueling_dqn_video.mp4'))

    print(f"\nBest Scores:\nRandom Agent: {random_score}\nOriginal DQN: {original_score}\nDueling DQN: {dueling_score}")
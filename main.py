import argparse
import yaml
from data import DataLoader
from envs import TradingEnv, Portfolio
from agents import RLAgent, SolverAgent, MarketObserver
from utils import Metrics
import numpy as np
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser(description="Reinforcement Learning for Quantitative Investment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: Load Market Data
    loader = DataLoader()
    data_source = config['data']['source']
    
    if data_source == 'csv':
        data = loader.load_data_from_csv(config['data']['csv_path'])
    elif data_source == 'yfinance':
        data = loader.load_data_from_yfinance(config['data']['tickers'],
                                              config['data']['start_date'],
                                              config['data']['end_date'])
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    processed_data = loader.process_data(data, columns_to_normalize=['Close'])

    # Step 2: Split Data into Train, Validation, and Test
    train_data, val_data, test_data = loader.split_data(processed_data,
                                                        train_ratio=config['train']['train_split'],
                                                        val_ratio=config['train']['val_split'])

    # Step 3: Initialize Environment, Portfolio, Agents, and Observer
    env = TradingEnv(data=train_data.values, 
                     initial_balance=config['env']['initial_balance'], 
                     transaction_cost=config['env']['transaction_cost'])
    
    portfolio = Portfolio(initial_balance=config['env']['initial_balance'])

    # RL Agent Initialization with flexible options
    rl_agent = RLAgent(state_dim=train_data.shape[1], action_dim=train_data.shape[1], max_action=1, 
                       gamma=config['agent']['gamma'], tau=config['agent']['tau'], device=device)
    
    # Solver Agent for risk minimization
    solver_agent = SolverAgent(n_assets=train_data.shape[1], risk_function=lambda w: np.std(w))  # Example risk function
    
    # Market Observer Model Selection
    market_observer = MarketObserver(model_type=config['agent']['market_observer'], 
                                     input_dim=train_data.shape[1], device=device)

    # Step 4: Training Loop with adjustable number of episodes
    num_episodes = config['train']['num_episodes']
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(len(train_data) - 1):
            # Step 4.1: Use RL Agent to choose action (new portfolio weights)
            #print('=== RL Agent choosing action ===')
            action = rl_agent.choose_action(state)
            
            # Step 4.2: Rebalance portfolio based on RL agent's action
            #print('=== Rebalancing portfolio ===')
            portfolio_value = portfolio.rebalance(action, state)

            # Step 4.3: Take a step in the environment
            #print('=== Taking a step ===')
            next_state, reward, done, _ = env.step(action)
            
            # Step 4.4: Store transition in replay buffer
            #print('=== Storing transition ===')
            rl_agent.store_transition(state, action, reward, next_state, done)

            # Step 4.5: Update RL agent using experience from the replay buffer
            #print('=== Updating policy ===')
            rl_agent.update_policy()

            # # Step 4.5: Use Solver Agent to adjust portfolio
            # optimized_weights = solver_agent.optimize_risk(action)
            # action = solver_agent.adjust_portfolio(portfolio, optimized_weights)

            # # Step 4.6: Predict market trends with Market Observer
            # market_trend = market_observer.predict_trends(state)
            
            # Step 4.7: Accumulate total reward
            total_reward += reward
            state = next_state
            # input()
            
            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    # Step 5: Evaluate on Validation Set (similar loop but without training)
    env = TradingEnv(data=val_data.values)
    state = env.reset()
    portfolio_values = [portfolio.initial_balance]

    for step in range(len(val_data) - 1):
        action = rl_agent.choose_action(state)
        portfolio_value = portfolio.rebalance(action, state)
        next_state, reward, done, _ = env.step(action)
        portfolio_values.append(portfolio_value)
        state = next_state

    # Step 6: Compute Performance Metrics
    portfolio_values = np.array(portfolio_values)
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    annual_ret = Metrics.annual_return(portfolio_values)
    max_dd = Metrics.maximum_drawdown(portfolio_values)
    sharpe = Metrics.sharpe_ratio(portfolio_returns)
    short_term_risk = Metrics.short_term_risk(portfolio_returns)

    print(f"Annual Return: {annual_ret:.4f}")
    print(f"Maximum Drawdown: {max_dd:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Short-term Risk: {short_term_risk:.4f}")

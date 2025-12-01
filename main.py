import yaml
import os
from trader.envs.factory import EnvironmentFactory
from trader.trainer import Trainer
from trader.utils.seed import SeedManager
import numpy as np

def load_config(config_path: str = './configs/defaults.yaml') -> dict:
    """加載配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"[Main] Loaded configuration from {config_path}\n")
    return config

def main():
    """主程序"""
    config = load_config('./configs/defaults.yaml')
    
    # ← 驗證必需的鍵
    required_keys = {
        'data': ['ticker_list', 'date_start', 'date_end'],
        'env': ['initial_balance', 'max_steps', 'k', 'transaction_cost'],
        'training': ['algorithm', 'max_episodes', 'update_frequency', 'model_type'],
        'hyperparameters': ['actor_lr', 'critic_lr', 'gamma', 'hidden_dim', 'batch_size'],
        'evaluation': ['num_episodes']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise KeyError(f"Missing configuration section: {section}")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing key '{key}' in section '{section}'")
    
    data_cfg = config['data']
    env_cfg = config['env']
    train_cfg = config['training']
    hyper_cfg = config['hyperparameters']
    eval_cfg = config['evaluation']
    
    # ← 獲取種子（默認 42）
    seed = config.get('seed', 42)
    
    # ← 設置全局隨機種子
    SeedManager.set_seed(seed)
    
    stock_symbols = data_cfg['ticker_list']
    algorithm = train_cfg['algorithm']
    
    print(f"[Main] Algorithm: {algorithm.upper()}")
    print(f"[Main] Stocks: {len(stock_symbols)}")
    print(f"[Main] Random seed: {seed}\n")
    
    # 創建環境
    env = EnvironmentFactory.create_trading_env({
        'num_stocks': len(stock_symbols),
        'stock_symbols': stock_symbols,
        'initial_balance': env_cfg['initial_balance'],
        'max_steps': env_cfg['max_steps'],
        'start_date': data_cfg['date_start'],
        'end_date': data_cfg['date_end'],
        'k': env_cfg['k'],
        'transaction_cost': env_cfg['transaction_cost'],
        'seed': seed  # ← 傳遞種子
    })
    
    # 創建訓練器
    trainer = Trainer(
        env=env,
        algorithm=algorithm,
        max_episodes=train_cfg['max_episodes'],
        max_timesteps=train_cfg.get('max_timesteps', 50000),
        update_frequency=train_cfg['update_frequency'],
        model_type=train_cfg['model_type'],
        seed=seed,  # ← 傳遞種子
        k=env_cfg['k'],
        actor_lr=hyper_cfg['actor_lr'],
        critic_lr=hyper_cfg['critic_lr'],
        gamma=hyper_cfg['gamma'],
        hidden_dim=hyper_cfg['hidden_dim'],
        batch_size=hyper_cfg['batch_size']
    )
    
    # 訓練
    train_results = trainer.train()
    
    # 保存模型
    model_path = f"./models/{algorithm}_agent.pth"
    os.makedirs('./models', exist_ok=True)
    trainer.save_model(model_path)
    
    # 評估
    eval_results = trainer.evaluate(
        num_episodes=config['evaluation']['num_episodes'],
        deterministic_seed=True  # ← 使用確定性種子確保可重現
    )

if __name__ == "__main__":
    main()
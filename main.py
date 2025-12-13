import yaml
import os
import argparse
from trader.envs.factory import EnvironmentFactory
from trader.trainer import Trainer, HierarchicalTrainer
from trader.utils.seed import SeedManager

def load_config(config_path: str = './configs/defaults.yaml') -> dict:
    """加載配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"[Main] Loaded configuration from {config_path}\n")
    return config

def get_date_ranges(data_cfg: dict) -> tuple:
    """
    從配置中提取訓練和測試的日期範圍
    
    :param data_cfg: 數據配置字典
    :return: (train_start, train_end, test_start, test_end) 元組
    """
    train_start = data_cfg.get('train_date_start', data_cfg.get('date_start', '2010-01-01'))
    train_end = data_cfg.get('train_date_end', data_cfg.get('date_end', '2021-09-30'))
    test_start = data_cfg.get('test_date_start', data_cfg.get('date_start', '2021-10-01'))
    test_end = data_cfg.get('test_date_end', data_cfg.get('date_end', '2023-03-01'))
    
    return train_start, train_end, test_start, test_end

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='Multi-Stock Deep Reinforcement Learning Trader')
    
    parser.add_argument('--config', '-c', type=str, default='./configs/defaults.yaml',
                       help='配置檔案路徑 (default: ./configs/defaults.yaml)')
    parser.add_argument('--train', action='store_true',
                       help='執行訓練模式')
    parser.add_argument('--eval', action='store_true',
                       help='執行評估模式')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='模型檔案路徑 (評估時使用)')
    parser.add_argument('--seed', type=int, default=None,
                       help='隨機種子 (覆蓋配置檔案設定)')
    parser.add_argument('--num-workers', '-w', type=int, default=None,
                       help='平行訓練的 worker 數量 (預設: Sub-Agent 數量)')
    parser.add_argument('--train-sub-agent', type=int, default=None,
                       help='訓練指定索引的 Sub-Agent (0, 1, 2, ...)')
    parser.add_argument('--train-final-only', action='store_true',
                       help='只訓練 Final Agent (使用已訓練的 Sub-Agent 模型)')
    
    return parser.parse_args()

def main():
    """主程序"""
    args = parse_args()
    config = load_config(args.config)
    
    # 根據命令列參數決定操作模式
    if args.train:
        config['agent_mode']['operation'] = 'training'
    elif args.eval:
        config['agent_mode']['operation'] = 'evaluation'
    else:
        # 預設為訓練模式
        config['agent_mode']['operation'] = 'training'
    
    # 如果命令列指定了種子，覆蓋配置檔案
    if args.seed is not None:
        config['seed'] = args.seed
    
    # ← 驗證必需的鍵
    required_keys = {
        'data': ['ticker_list', 'date_start', 'date_end'],
        'env': ['initial_balance', 'max_steps', 'transaction_cost'],
        'training': ['max_episodes', 'update_frequency'],
        'hyperparameters': ['actor_lr', 'critic_lr', 'gamma', 'hidden_dim', 'batch_size'],
        'evaluation': ['num_episodes'], 
        'agent_mode': ['mode']
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
    agent_mode_cfg = config['agent_mode']
    
    # ★★★ 新增：提取訓練和測試的日期範圍 ★★★
    train_start, train_end, test_start, test_end = get_date_ranges(data_cfg)
    
    print(f"[Main] 📅 日期範圍:")
    print(f"  - 訓練期間: {train_start} 至 {train_end}")
    print(f"  - 測試期間: {test_start} 至 {test_end}\n")
    
    # ← 確保超參數類型正確
    actor_lr = float(hyper_cfg['actor_lr'])
    critic_lr = float(hyper_cfg['critic_lr'])
    gamma = float(hyper_cfg['gamma'])
    hidden_dim = int(hyper_cfg['hidden_dim'])
    batch_size = int(hyper_cfg['batch_size'])
    
    # ← 獲取種子（默認 42）
    seed = config.get('seed', 42)
    
    # ← 設置全局隨機種子
    SeedManager.set_seed(seed)
    
    stock_symbols = data_cfg['ticker_list']
    
    # 獲取操作模式
    operation = agent_mode_cfg.get('operation', 'training')

    print(f"\n{'='*70}")
    print(f"[Main] 🚀 Multi-Stock Deep Reinforcement Learning Trader")
    print(f"{'='*70}")
    print(f"[Main] Agent Mode: {agent_mode_cfg['mode'].upper()}")
    print(f"[Main] Operation: {operation.upper()}")
    print(f"[Main] Stocks: {len(stock_symbols)}")
    print(f"[Main] Random seed: {seed}")
    print(f"[Main] Max Episodes: {train_cfg['max_episodes']}\n")
    
    # 檢查是否訓練單個 Sub-Agent
    if args.train_sub_agent is not None:
        print(f"\n{'='*70}")
        print(f"[Main] 🚀 訓練單個 Sub-Agent")
        print(f"{'='*70}\n")
        
        sub_agents_cfg = agent_mode_cfg.get('sub_agents', [])
        
        if args.train_sub_agent < 0 or args.train_sub_agent >= len(sub_agents_cfg):
            print(f"❌ 無效的 Sub-Agent 索引: {args.train_sub_agent}")
            print(f"可用索引: 0-{len(sub_agents_cfg)-1}")
            return
        
        sub_agent_cfg = sub_agents_cfg[args.train_sub_agent]
        agent_name = sub_agent_cfg.get('name', f'Sub-Agent-{args.train_sub_agent}')
        
        print(f"[Main] Sub-Agent 索引: {args.train_sub_agent}")
        print(f"[Main] 名稱: {agent_name}")
        print(f"[Main] 演算法: {sub_agent_cfg.get('algorithm', 'a2c').upper()}")
        print(f"[Main] 模型: {sub_agent_cfg.get('model_type', 'mlp').upper()}\n")
        
        # ★★★ 修改：使用訓練日期範圍 ★★★
        # 創建訓練環境
        train_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': train_start,
            'end_date': train_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed,
            'agent_type': sub_agent_cfg.get('agent_type', 'direction'),
            'model_type': sub_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })
        
        # 創建測試環境
        test_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': test_start,
            'end_date': test_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed + 1,
            'agent_type': sub_agent_cfg.get('agent_type', 'direction'),
            'model_type': sub_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })
        
        # 創建訓練器（Sub-Agent 不使用注意力機制）
        trainer = Trainer(
            agent_name=agent_name,
            env=train_env,  # ★★★ 使用訓練環境
            algorithm=sub_agent_cfg.get('algorithm', 'a2c'),
            max_episodes=train_cfg['max_episodes'],
            update_frequency=train_cfg['update_frequency'],
            model_type=sub_agent_cfg.get('model_type', 'mlp'),
            seed=seed,
            agent_mode='single-agent',
            use_attention=False,  # Sub-Agent 不使用注意力機制
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )
        
        # 手動設置測試環境
        trainer.test_env = test_env  # ★★★ 設置測試環境
        
        # 訓練
        print(f"[Main] ✅ 開始訓練 Sub-Agent: {agent_name}\n")
        trainer.train()
        
        # 儲存模型
        os.makedirs('./models/sub_agents', exist_ok=True)
        model_path = f"./models/sub_agents/{agent_name}_agent.pth"
        trainer.save_model(model_path)
        print(f"\n[Main] ✅ Sub-Agent 訓練完成！")
        print(f"[Main] 模型已保存到: {model_path}\n")
        
        return
    
    # 如果只訓練 Final Agent
    if args.train_final_only:
        print(f"\n{'='*70}")
        print(f"[Main] 🎯 訓練 Final Agent (使用已訓練的 Sub-Agent 模型)")
        print(f"{'='*70}\n")
        
        from trader.parallel_trainer import SubAgentEnsemble
        from trader.envs.final_agent_env import FinalAgentEnv
        
        sub_agents_cfg = agent_mode_cfg.get('sub_agents', [])
        final_agent_cfg = agent_mode_cfg.get('final_agent', {})
        
        # 準備 Sub-Agent 模型路徑
        model_paths = {}
        
        # ★★★ 修改：使用訓練日期創建臨時環境獲取維度 ★★★
        temp_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': train_start,
            'end_date': train_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed,
            'model_type': final_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })
        
        print(f"[Main] 檢查 Sub-Agent 模型...\n")
        all_found = True
        for i, sub_agent in enumerate(sub_agents_cfg):
            agent_name = sub_agent.get('name', f'Sub-Agent-{i}')
            model_path = f"./models/sub_agents/{agent_name}_agent.pth"
            
            if os.path.exists(model_path):
                model_paths[agent_name] = {
                    'path': model_path,
                    'algorithm': sub_agent.get('algorithm', 'a2c'),
                    'model_type': sub_agent.get('model_type', 'mlp'),
                    'state_dim': temp_env.state_dim,
                    'action_dim': temp_env.action_dim,
                    'hidden_dim': hidden_dim,
                }
                size = os.path.getsize(model_path) / 1024 / 1024
                print(f"  ✓ [{i}] {agent_name}: {size:.2f} MB")
            else:
                print(f"  ✗ [{i}] {agent_name}: NOT FOUND at {model_path}")
                all_found = False
        
        if not all_found:
            print(f"\n❌ 缺少一些 Sub-Agent 模型，請先訓練所有 Sub-Agents")
            print(f"執行以下命令:")
            print(f"  ./run_pipeline.sh train-parallel")
            return
        
        print(f"\n✓ 所有 Sub-Agent 模型已找到\n")
        
        # 建立 Sub-Agent 集成器
        print(f"[Main] 建立 Sub-Agent 集成器...\n")
        ensemble = SubAgentEnsemble(model_paths)
        
        # ★★★ 修改：使用訓練日期創建訓練環境 ★★★
        # 創建 Final Agent 訓練環境
        train_base_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': train_start,
            'end_date': train_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed + 100,
            'agent_type': final_agent_cfg.get('agent_type', 'final'),
            'model_type': final_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })
        
        train_final_env = FinalAgentEnv(train_base_env, ensemble)
        
        # ★★★ 修改：使用測試日期創建測試環境 ★★★
        # 創建 Final Agent 測試環境
        test_base_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': test_start,
            'end_date': test_end,
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed + 101,
            'agent_type': final_agent_cfg.get('agent_type', 'final'),
            'model_type': final_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })
        
        test_final_env = FinalAgentEnv(test_base_env, ensemble)
        
        # 提取 Final Agent 注意力參數
        use_attention = final_agent_cfg.get('use_attention', False)
        num_heads = final_agent_cfg.get('num_heads', 4)
        attention_type = final_agent_cfg.get('attention_type', 'simple')
        
        print(f"[Main] Final Agent 配置:")
        print(f"  - 演算法: {final_agent_cfg.get('algorithm', 'ddpg').upper()}")
        print(f"  - 模型: {final_agent_cfg.get('model_type', 'mlp').upper()}")
        print(f"  - 注意力: {'✓ 啟用' if use_attention else '✗ 禁用'}")
        if use_attention:
            print(f"    - 類型: {attention_type}")
            print(f"    - 頭數: {num_heads}")
        print()
        
        # 創建 Final Agent Trainer
        final_trainer = Trainer(
            agent_name=final_agent_cfg.get('name', 'Final_Agent'),
            env=train_final_env,  # ★★★ 使用訓練環境
            algorithm=final_agent_cfg.get('algorithm', 'ddpg'),
            max_episodes=train_cfg['max_episodes'],
            update_frequency=train_cfg['update_frequency'],
            model_type=final_agent_cfg.get('model_type', 'mlp'),
            seed=seed + 100,
            agent_mode='single-agent',
            use_attention=use_attention,
            num_heads=num_heads,
            attention_type=attention_type,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )
        
        # 手動設置測試環境
        final_trainer.test_env = test_final_env  # ★★★ 設置測試環境
        
        # 訓練 Final Agent
        print(f"[Main] ✅ 開始訓練 Final Agent\n")
        final_trainer.train()
        
        # 儲存模型
        os.makedirs('./models', exist_ok=True)
        final_model_path = f"./models/{final_agent_cfg.get('name', 'Final_Agent')}_agent.pth"
        final_trainer.save_model(final_model_path)
        
        print(f"\n[Main] ✅ Final Agent 訓練完成！")
        print(f"[Main] 模型已保存到: {final_model_path}\n")
        
        return

    # ========== Multi-Agent 模式：使用 HierarchicalTrainer 平行訓練 ==========
    if agent_mode_cfg['mode'] == 'multi-agent':
        sub_agents_cfg = agent_mode_cfg.get('sub_agents', [])
        final_agent_cfg = agent_mode_cfg.get('final_agent', {})
        num_sub_agents = len(sub_agents_cfg)
        
        print(f"[Main] Number of Sub-Agents: {num_sub_agents}")
        print(f"[Main] 🚀 使用平行訓練模式 (HierarchicalTrainer)\n")
        
        # 顯示每個 Sub-Agent 的配置
        print(f"[Sub-Agents Configuration]:")
        for i, sub_agent in enumerate(sub_agents_cfg):
            use_attn = sub_agent.get('use_attention', False)
            print(f"  [{i+1}] Name: {sub_agent.get('name', 'Unknown')}")
            print(f"      Algorithm: {sub_agent.get('algorithm', 'N/A').upper()}")
            print(f"      Model Type: {sub_agent.get('model_type', 'N/A').upper()}")
            print(f"      Agent Type: {sub_agent.get('agent_type', 'N/A')}")
            print(f"      Use Attention: {use_attn}")
        
        # 顯示 Final Agent 的配置
        use_attn = final_agent_cfg.get('use_attention', False)
        attn_type = final_agent_cfg.get('attention_type', 'N/A') if use_attn else 'N/A'
        num_heads = final_agent_cfg.get('num_heads', 'N/A') if use_attn else 'N/A'
        print(f"\n[Final Agent Configuration]:")
        print(f"  Name: {final_agent_cfg.get('name', 'Unknown')}")
        print(f"  Algorithm: {final_agent_cfg.get('algorithm', 'N/A').upper()}")
        print(f"  Model Type: {final_agent_cfg.get('model_type', 'N/A').upper()}")
        print(f"  Agent Type: {final_agent_cfg.get('agent_type', 'N/A')}")
        print(f"  Use Attention: {use_attn}")
        if use_attn:
            print(f"  Attention Type: {attn_type}")
            print(f"  Attention Heads: {num_heads}\n")
        else:
            print()
        
        # 創建 HierarchicalTrainer（平行訓練）
        hierarchical_trainer = HierarchicalTrainer(config, seed=seed)
        
        if operation == 'training':
            print(f"[Main] ✅ Starting Multi-Agent Parallel Training...\n")
            
            # 獲取 worker 數量（命令列參數優先）
            num_workers = args.num_workers or agent_mode_cfg.get('num_workers', None)
            
            # 執行平行訓練
            hierarchical_trainer.train(num_workers=num_workers)
            
            # 儲存所有模型
            hierarchical_trainer.save_all_models('./models')
            
            print(f"\n[Main] ✅ Multi-Agent Training Complete!\n")
            
        elif operation == 'evaluation':
            print(f"[Main] ✅ Starting Multi-Agent Evaluation...\n")
            
            # 評估
            eval_results = hierarchical_trainer.evaluate(deterministic_seed=True)
            
    
    # ========== Single-Agent 模式 ==========
    else:
        print(f"[Main] Agent Mode: Single-Agent\n")
        
        # 從 agent_mode 配置讀取演算法
        algorithm = agent_mode_cfg.get('final_agent_algorithm', 'ddpg')
        model_type = agent_mode_cfg.get('final_agent_model_type', 'mlp')
        agent_name = agent_mode_cfg.get('final_agent_name', 'Final_Agent')
        use_attention = agent_mode_cfg.get('use_attention', False)
        num_heads = agent_mode_cfg.get('num_heads', 4)
        attention_type = agent_mode_cfg.get('attention_type', 'simple')
        
        print(f"[Main] Algorithm: {algorithm.upper()}")
        print(f"[Main] Model Type: {model_type.upper()}")
        print(f"[Main] Use Attention: {use_attention}")
        if use_attention:
            print(f"[Main] Attention Type: {attention_type}")
            print(f"[Main] Attention Heads: {num_heads}")
        print()
        
        # ★★★ 修改：使用訓練日期創建訓練環境 ★★★
        # 創建訓練環境
        train_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': train_start,
            'end_date': train_end,
            'k': env_cfg.get('k', 1),
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed,
            'agent_type': sub_agent_cfg.get('agent_type', 'direction'),
            'model_type': sub_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })
        
        # ★★★ 修改：使用測試日期創建測試環境 ★★★
        # 創建測試環境
        test_env = EnvironmentFactory.create_trading_env({
            'num_stocks': len(stock_symbols),
            'stock_symbols': stock_symbols,
            'initial_balance': env_cfg['initial_balance'],
            'max_steps': env_cfg['max_steps'],
            'start_date': test_start,
            'end_date': test_end,
            'k': env_cfg.get('k', 1),
            'transaction_cost': env_cfg['transaction_cost'],
            'seed': seed + 1,
            'agent_type': sub_agent_cfg.get('agent_type', 'direction'),
            'model_type': sub_agent_cfg.get('model_type', 'mlp'),
            'window_size': env_cfg.get('window_size', 10)
        })

        # 創建訓練器
        trainer = Trainer(
            agent_name=agent_name,
            env=train_env,  # ★★★ 使用訓練環境
            algorithm=algorithm,
            max_episodes=train_cfg['max_episodes'],
            max_timesteps=train_cfg.get('max_timesteps', 50000),
            update_frequency=train_cfg['update_frequency'],
            model_type=model_type,
            seed=seed,
            agent_mode='single-agent',
            use_attention=use_attention,
            num_heads=num_heads,
            attention_type=attention_type,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
        )
        
        # 手動設置測試環境
        trainer.test_env = test_env  # ★★★ 設置測試環境
        
        if operation == 'training':
            print(f"[Main] ✅ Starting Single-Agent Training...\n")
            trainer.train()

            os.makedirs('./models', exist_ok=True)
            trainer.save_model(f"./models/{agent_name}.pth")
            print(f"\n[Main] Model saved to ./models/{agent_name}.pth\n")
            
        elif operation == 'evaluation':
            print(f"[Main] ✅ Starting Single-Agent Evaluation...\n")
            
            # 載入模型
            model_path = args.model or f"./models/{agent_name}.pth"
            if os.path.exists(model_path):
                trainer.load_model(model_path)
                print(f"[Main] Loaded model from {model_path}\n")
            else:
                print(f"[Main] Warning: Model not found at {model_path}, using untrained model\n")
            
            # 評估
            eval_results = trainer.evaluate(deterministic_seed=True)

if __name__ == "__main__":
    main()
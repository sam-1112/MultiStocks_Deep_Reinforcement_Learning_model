import argparse
import os
import yaml
import torch
import numpy as np
from tests.backtest import BacktestEngine, BacktestComparator
from trader.envs.factory import EnvironmentFactory
from trader.envs.final_agent_env import FinalAgentEnv
from trader.utils.seed import SeedManager
from trader.parallel_trainer import SubAgentEnsemble


def load_config(config_path: str = './configs/defaults.yaml') -> dict:
    """加載配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def infer_state_dim_from_checkpoint(checkpoint: dict, algo: str) -> int:
    """
    從模型檢查點推斷狀態維度
    
    :param checkpoint: 模型檢查點字典
    :param algo: 演算法名稱
    :return: 推斷的狀態維度
    """
    try:
        if algo.lower() == 'ddqn':
            # DDQN: 從 q_network 的第一層推斷
            q_network_state = checkpoint.get('q_network', {})
            # 查找第一層的權重
            for key in sorted(q_network_state.keys()):
                param = q_network_state[key]
                # 第一層應該是 state_encoder 或直接的線性層
                if 'weight' in key and param.dim() == 2:
                    # weight shape: [out_features, in_features]
                    # in_features 是狀態維度
                    state_dim = param.shape[1]
                    print(f"  ├─ 從 {key} 推斷狀態維度: {state_dim}")
                    return state_dim
        
        elif algo.lower() == 'ddpg':
            # DDPG: 從 actor 的第一層推斷
            actor_state = checkpoint.get('actor', {})
            for key in sorted(actor_state.keys()):
                param = actor_state[key]
                if 'weight' in key and param.dim() == 2:
                    state_dim = param.shape[1]
                    print(f"  ├─ 從 {key} 推斷狀態維度: {state_dim}")
                    return state_dim
        
        elif algo.lower() == 'a2c':
            # A2C: 從 actor 的第一層推斷
            actor_state = checkpoint.get('actor', {})
            for key in sorted(actor_state.keys()):
                param = actor_state[key]
                if 'weight' in key and param.dim() == 2:
                    state_dim = param.shape[1]
                    print(f"  ├─ 從 {key} 推斷狀態維度: {state_dim}")
                    return state_dim
    
    except Exception as e:
        print(f"  ⚠️  無法推斷狀態維度: {str(e)}")
    
    return None


def detect_model_config(model_path: str, algo: str) -> dict:
    """
    檢測模型配置（是否使用注意力機制等）
    
    :param model_path: 模型檔案路徑
    :param algo: 演算法名稱
    :return: 模型配置字典
    """
    print(f"[Backtest] 檢測模型配置: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # ← 根據演算法檢查注意力層
        if algo.lower() == 'ddqn':
            # DDQN: 檢查 q_network 中的注意力
            q_network = checkpoint.get('q_network', {})
            has_attention = any('attention' in key for key in q_network.keys())
        
        elif algo.lower() == 'a2c':
            # A2C: 檢查 actor 中的注意力
            actor = checkpoint.get('actor', {})
            has_attention = any('attention' in key for key in actor.keys())
        
        elif algo.lower() == 'ddpg':
            # DDPG: 檢查 actor 中的注意力
            actor = checkpoint.get('actor', {})
            has_attention = any('attention' in key for key in actor.keys())
        
        else:
            has_attention = False
        
        # ← 新增：從模型推斷狀態維度
        state_dim = infer_state_dim_from_checkpoint(checkpoint, algo)
        
        # 提取配置
        config = {
            'use_attention': has_attention,
            'attention_type': checkpoint.get('attention_type', 'simple'),
            'state_dim': state_dim,
        }
        
        print(f"  ├─ 演算法: {algo.upper()}")
        print(f"  ├─ 使用注意力: {config['use_attention']}")
        print(f"  ├─ 注意力類型: {config['attention_type']}")
        if state_dim is not None:
            print(f"  ├─ 推斷狀態維度: {state_dim}")
        print(f"  └─ ✓ 檢測完成\n")
        
        return config
    
    except Exception as e:
        print(f"  ⚠️  無法自動檢測配置: {str(e)}")
        print(f"  └─ 使用預設配置\n")
        return {'use_attention': False, 'attention_type': 'simple', 'state_dim': None}


def backtest_single_subagent(agent_name: str, model_path: str, algo: str, config: dict,
                             use_attention: bool = False, attention_type: str = 'simple',
                             inferred_state_dim: int = None, model_type: str = 'mlp'):  # ← 添加參數
    """
    回測單個 Sub-Agent
    
    :param agent_name: 代理名稱
    :param model_path: 模型路徑
    :param algo: 演算法名稱
    :param config: 配置字典
    :param use_attention: 是否使用注意力機制
    :param attention_type: 注意力類型
    :param inferred_state_dim: 推斷的狀態維度
    :param model_type: 模型類型
    """
    print(f"\n{'='*70}")
    print(f"[Backtest] 回測 Sub-Agent: {agent_name}")
    print(f"{'='*70}\n")
    
    # 解析配置
    data_cfg = config['data']
    env_cfg = config['env']
    hyper_cfg = config['hyperparameters']
    
    # 設置種子
    seed = config.get('seed', 42)
    SeedManager.set_seed(seed)
    
    # 創建環境
    print(f"[Backtest] 創建環境...")
    env = EnvironmentFactory.create_trading_env({
        'num_stocks': len(data_cfg['ticker_list']),
        'stock_symbols': data_cfg['ticker_list'],
        'initial_balance': env_cfg['initial_balance'],
        'max_steps': env_cfg['max_steps'],
        'start_date': data_cfg['date_start'],
        'end_date': data_cfg['date_end'],
        'transaction_cost': env_cfg['transaction_cost'],
        'seed': seed,
    })
    
    # ← 使用推斷的 state_dim 而不是環境的 state_dim
    actual_state_dim = inferred_state_dim if inferred_state_dim is not None else env.state_dim
    
    print(f"  ├─ 股票數量: {env.action_dim}")
    print(f"  ├─ 狀態維度: {actual_state_dim}")
    if inferred_state_dim is not None and inferred_state_dim != env.state_dim:
        print(f"  │  (推斷自模型: {inferred_state_dim}, 環境計算: {env.state_dim})")
    print(f"  └─ ✓ 環境創建完成\n")
    
    # 檢查模型檔案
    if not os.path.exists(model_path):
        print(f"❌ 模型檔案不存在: {model_path}")
        return None
    
    # 初始化回測引擎
    backtest_engine = BacktestEngine(
        initial_balance=env_cfg['initial_balance'],
        transaction_cost=env_cfg['transaction_cost']
    )
    
    # 加載模型信息
    print(f"[Backtest] 加載模型...")
    model_info = backtest_engine.load_model(path=model_path)
    
    # 根據模型信息創建算法實例
    from trader.factory import AlgorithmFactory
    
    algo_type = model_info['algo_type']
    
    print(f"[Backtest] 創建算法實例...")
    
    # 創建算法實例 - 使用推斷的狀態維度和提供的模型類型
    algo_instance = AlgorithmFactory.create(
        algorithm=algo_type,
        state_dim=actual_state_dim,
        action_dim=env.action_dim,
        hidden_dim=int(hyper_cfg.get('hidden_dim', 256)),
        model_type=model_type,  # ← 添加
        use_attention=use_attention,
        attention_type=attention_type,
        num_heads=int(hyper_cfg.get('num_heads', 4)),
        configs=config,  # ← 新增：傳遞完整配置
    )
    
    # 加載權重到算法實例
    algo_instance.load_model(model_path)
    
    # 執行回測
    num_episodes = config.get('evaluation', {}).get('num_episodes', 5)
    print(f"\n[Backtest] 執行回測 ({num_episodes} 回合)...\n")
    results = backtest_engine.run_backtest(
        env=env,
        model=algo_instance,
        num_episodes=num_episodes,
        deterministic=True
    )
    
    # 計算性能指標
    metrics = backtest_engine.calculate_metrics(results)
    
    # 打印報告
    backtest_engine.print_report(metrics)
    
    # 保存報告
    backtest_engine.save_report(
        metrics, results,
        f'./backtest_reports/{agent_name}_metrics.json'
    )
    
    backtest_engine.generate_csv_report(
        results,
        f'./backtest_reports/{agent_name}_results.csv'
    )
    
    return metrics


def backtest_hierarchical_agents(config: dict):
    """
    回測階層式 RL（Final Agent + Sub-Agents）
    
    Final Agent 依賴 Sub-Agents 的 Q-values 進行決策
    """
    print(f"\n{'='*70}")
    print(f"[Backtest] 階層式 RL 回測 (Final Agent + Sub-Agents)")
    print(f"{'='*70}")
    
    comparator = BacktestComparator()
    
    # 解析配置
    data_cfg = config['data']
    env_cfg = config['env']
    hyper_cfg = config['hyperparameters']
    agent_mode_cfg = config['agent_mode']
    
    # 設置種子
    seed = config.get('seed', 42)
    SeedManager.set_seed(seed)
    
    # ========== 第一步：創建基礎環境 ==========
    print(f"\n[Step 1] 創建基礎交易環境...")
    print("─" * 70)
    
    base_env = EnvironmentFactory.create_trading_env({
        'num_stocks': len(data_cfg['ticker_list']),
        'stock_symbols': data_cfg['ticker_list'],
        'initial_balance': env_cfg['initial_balance'],
        'max_steps': env_cfg['max_steps'],
        'start_date': data_cfg['date_start'],
        'end_date': data_cfg['date_end'],
        'transaction_cost': env_cfg['transaction_cost'],
        'seed': seed,
    })
    
    print(f"  ├─ 股票數量: {base_env.action_dim}")
    print(f"  ├─ 基礎狀態維度: {base_env.state_dim}")
    print(f"  └─ ✓ 基礎環境創建完成\n")
    
    # ========== 第二步：加載 Sub-Agents ==========
    print(f"\n[Step 2] 加載 Sub-Agents...")
    print("─" * 70)
    
    sub_agents_cfg = agent_mode_cfg.get('sub_agents', [])
    sub_agent_models = {}
    sub_agent_configs = {}
    
    from trader.factory import AlgorithmFactory
    
    for i, sub_agent_cfg in enumerate(sub_agents_cfg):
        agent_name = sub_agent_cfg.get('name', f'Sub-Agent-{i}')
        model_path = f"./models/sub_agents/{agent_name}_agent.pth"
        algo = sub_agent_cfg.get('algorithm', 'a2c')
        model_type = sub_agent_cfg.get('model_type', 'mlp')  # ← 從配置取得
        
        print(f"\n  [{i+1}/{len(sub_agents_cfg)}] {agent_name}")
        
        # 檢測模型配置
        model_cfg = detect_model_config(model_path, algo)
        
        if not os.path.exists(model_path):
            print(f"  ❌ 模型檔案不存在: {model_path}")
            continue
        
        # 推斷狀態維度
        actual_state_dim = model_cfg.get('state_dim') or base_env.state_dim
        
        # 創建算法實例 - ← 添加 model_type
        algo_instance = AlgorithmFactory.create(
            algorithm=algo,
            state_dim=actual_state_dim,
            action_dim=base_env.action_dim,
            hidden_dim=int(hyper_cfg.get('hidden_dim', 256)),
            model_type=model_type,
            use_attention=model_cfg['use_attention'],
            attention_type=model_cfg['attention_type'],
            num_heads=int(hyper_cfg.get('num_heads', 4)),
            configs=config,  # ← 新增：傳遞完整配置
        )
        
        # 加載權重
        algo_instance.load_model(model_path)
        
        sub_agent_models[agent_name] = algo_instance
        sub_agent_configs[agent_name] = {
            'algo': algo,
            'state_dim': actual_state_dim,
            'model_type': model_type,  # ← 新增
        }
        
        print(f"  ├─ 狀態維度: {actual_state_dim}")
        print(f"  ├─ 模型類型: {model_type}")
        print(f"  ├─ 使用注意力: {model_cfg['use_attention']}")
        print(f"  └─ ✓ {agent_name} 已加載\n")
    
    if not sub_agent_models:
        print(f"\n❌ 沒有可用的 Sub-Agent 模型")
        return
    
    # ========== 第三步：創建 Sub-Agent 集成 ==========
    print(f"\n[Step 3] 創建 Sub-Agent 集成...")
    print("─" * 70)
    
    try:
        ensemble = SubAgentEnsemble(sub_agent_models)
        print(f"  ├─ 已集成 {len(sub_agent_models)} 個 Sub-Agent")
        print(f"  └─ ✓ 集成創建完成\n")
    except Exception as e:
        print(f"  ❌ 無法創建集成: {str(e)}")
        return
    
    # ========== 第四步：創建 Final Agent 環境 ==========
    print(f"\n[Step 4] 創建 Final Agent 環境...")
    print("─" * 70)
    
    try:
        final_env = FinalAgentEnv(base_env, ensemble)
        print(f"  ├─ 基礎狀態維度: {final_env.base_state_dim}")
        print(f"  ├─ Q-values 維度: {final_env._q_values_dim}")
        print(f"  ├─ 最終狀態維度: {final_env.state_dim}")
        print(f"  └─ ✓ Final Agent 環境創建完成\n")
    except Exception as e:
        print(f"  ❌ 無法創建 Final Agent 環境: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 第五步：回測 Final Agent ==========
    print(f"\n[Step 5] 回測 Final Agent...")
    print("─" * 70)
    
    final_agent_cfg = agent_mode_cfg.get('final_agent', {})
    final_agent_name = final_agent_cfg.get('name', 'Final_Agent')
    final_model_path = f"./models/{final_agent_name}_agent.pth"
    final_algo = final_agent_cfg.get('algorithm', 'ddqn')
    final_model_type = final_agent_cfg.get('model_type', 'mlp')  # ← 從配置取得
    
    # 檢測 Final Agent 模型配置
    final_model_cfg = detect_model_config(final_model_path, final_algo)
    
    if not os.path.exists(final_model_path):
        print(f"❌ Final Agent 模型檔案不存在: {final_model_path}")
        return
    
    # 推斷 Final Agent 狀態維度
    final_actual_state_dim = final_model_cfg.get('state_dim') or final_env.state_dim
    
    # 創建 Final Agent 算法實例 - ← 添加 model_type
    from trader.factory import AlgorithmFactory
    
    final_algo_instance = AlgorithmFactory.create(
        algorithm=final_algo,
        state_dim=final_actual_state_dim,
        action_dim=final_env.action_dim,
        hidden_dim=int(hyper_cfg.get('hidden_dim', 256)),
        model_type=final_model_type,
        use_attention=final_model_cfg['use_attention'],
        attention_type=final_model_cfg['attention_type'],
        num_heads=int(hyper_cfg.get('num_heads', 4)),
        configs=config,  # ← 新增：傳遞完整配置
    )
    
    # 加載 Final Agent 權重
    final_algo_instance.load_model(final_model_path)
    
    print(f"  ├─ Final Agent 狀態維度: {final_actual_state_dim}")
    print(f"  ├─ 模型類型: {final_model_type}")
    print(f"  ├─ 使用注意力: {final_model_cfg['use_attention']}")
    print(f"  └─ ✓ Final Agent 已加載\n")
    
    # 執行階層式回測
    backtest_engine = BacktestEngine(
        initial_balance=env_cfg['initial_balance'],
        transaction_cost=env_cfg['transaction_cost']
    )
    
    num_episodes = config.get('evaluation', {}).get('num_episodes', 5)
    print(f"\n[Backtest] 執行階層式回測 ({num_episodes} 回合)...\n")
    
    results = backtest_engine.run_backtest(
        env=final_env,
        model=final_algo_instance,
        num_episodes=num_episodes,
        deterministic=True
    )
    
    # 計算性能指標
    metrics = backtest_engine.calculate_metrics(results)
    
    # 打印報告
    backtest_engine.print_report(metrics)
    
    # 保存報告
    backtest_engine.save_report(
        metrics, results,
        f'./backtest_reports/Hierarchical_RL_metrics.json'
    )
    
    backtest_engine.generate_csv_report(
        results,
        f'./backtest_reports/Hierarchical_RL_results.csv'
    )
    
    comparator.add_result('Hierarchical_RL', metrics)
    
    # ========== 第六步：回測個別 Sub-Agents（作為對照） ==========
    print(f"\n[Step 6] 回測個別 Sub-Agents（對照組）...")
    print("─" * 70)
    
    for i, sub_agent_cfg in enumerate(sub_agents_cfg):
        agent_name = sub_agent_cfg.get('name', f'Sub-Agent-{i}')
        model_path = f"./models/sub_agents/{agent_name}_agent.pth"
        algo = sub_agent_cfg.get('algorithm', 'a2c')
        model_type = sub_agent_cfg.get('model_type', 'mlp')  # ← 從配置取得
        
        model_cfg = detect_model_config(model_path, algo)
        
        metrics = backtest_single_subagent(
            agent_name=agent_name,
            model_path=model_path,
            algo=algo,
            config=config,
            use_attention=model_cfg['use_attention'],
            attention_type=model_cfg['attention_type'],
            inferred_state_dim=model_cfg.get('state_dim'),
            model_type=model_type,  # ← 傳遞
        )
        
        if metrics:
            comparator.add_result(agent_name, metrics)
    
    # ========== 第七步：輸出比較結果 ==========
    print(f"\n{'='*70}")
    print(f"[Backtest] 回測結果對比")
    print(f"{'='*70}\n")
    comparator.print_comparison()
    comparator.save_comparison('./backtest_reports/backtest_comparison.csv')


def backtest_subagents_only(config: dict):
    """
    回測所有 Sub-Agents（不使用階層式 RL）
    """
    print(f"\n{'='*70}")
    print(f"[Backtest] Sub-Agents 回測")
    print(f"{'='*70}")
    
    comparator = BacktestComparator()
    
    agent_mode_cfg = config['agent_mode']
    sub_agents_cfg = agent_mode_cfg.get('sub_agents', [])
    
    for i, sub_agent_cfg in enumerate(sub_agents_cfg):
        agent_name = sub_agent_cfg.get('name', f'Sub-Agent-{i}')
        model_path = f"./models/sub_agents/{agent_name}_agent.pth"
        algo = sub_agent_cfg.get('algorithm', 'a2c')
        model_type = sub_agent_cfg.get('model_type', 'mlp')  # ← 從配置取得
        
        model_cfg = detect_model_config(model_path, algo)
        
        metrics = backtest_single_subagent(
            agent_name=agent_name,
            model_path=model_path,
            algo=algo,
            config=config,
            use_attention=model_cfg['use_attention'],
            attention_type=model_cfg['attention_type'],
            inferred_state_dim=model_cfg.get('state_dim'),
            model_type=model_type,  # ← 傳遞
        )
        
        if metrics:
            comparator.add_result(agent_name, metrics)
    
    # 打印比較結果
    print(f"\n{'='*70}")
    print(f"[Backtest] 回測結果對比")
    print(f"{'='*70}\n")
    comparator.print_comparison()
    comparator.save_comparison('./backtest_reports/subagents_comparison.csv')


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description='回測強化學習交易模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 回測階層式 RL（Final Agent + Sub-Agents）
  python3 backtest_cli.py --hierarchical
  
  # 只回測 Sub-Agents
  python3 backtest_cli.py --subagents-only
  
  # 回測特定 Sub-Agent
  python3 backtest_cli.py --agent Direction_Agent
  
  # 指定回合數
  python3 backtest_cli.py --hierarchical --num-episodes 10
  
  # 指定配置檔案
  python3 backtest_cli.py --hierarchical --config ./configs/custom.yaml
        """
    )
    
    parser.add_argument('--hierarchical', action='store_true',
                       help='回測階層式 RL（Final Agent + Sub-Agents）')
    parser.add_argument('--subagents-only', action='store_true',
                       help='只回測所有 Sub-Agents')
    parser.add_argument('--agent', '-a', type=str, default=None,
                       help='回測指定的 Sub-Agent (e.g., Direction_Agent)')
    parser.add_argument('--config', '-c', type=str, default='./configs/defaults.yaml',
                       help='配置檔案路徑 (預設: ./configs/defaults.yaml)')
    parser.add_argument('--num-episodes', '-n', type=int, default=5,
                       help='回測的回合數 (預設: 5)')
    
    args = parser.parse_args()
    
    # 加載配置
    print(f"[Backtest] 加載配置: {args.config}\n")
    config = load_config(args.config)
    
    # 更新回合數
    if 'evaluation' not in config:
        config['evaluation'] = {}
    config['evaluation']['num_episodes'] = args.num_episodes
    
    # 創建報告目錄
    os.makedirs('./backtest_reports', exist_ok=True)
    
    try:
        # 執行回測
        if args.hierarchical:
            # 階層式 RL 回測
            backtest_hierarchical_agents(config)
        elif args.subagents_only:
            # 只回測 Sub-Agents
            backtest_subagents_only(config)
        elif args.agent:
            # 回測指定的 Sub-Agent
            agent_mode_cfg = config['agent_mode']
            sub_agents_cfg = agent_mode_cfg.get('sub_agents', [])
            
            agent_name = args.agent
            model_path = f"./models/sub_agents/{agent_name}_agent.pth"
            algo = 'a2c'
            model_type = 'mlp'  # 默認值
            
            # 從配置中查找演算法和模型類型
            for sub in sub_agents_cfg:
                if sub.get('name') == agent_name:
                    algo = sub.get('algorithm', 'a2c')
                    model_type = sub.get('model_type', 'mlp')  # ← 添加
                    break
            
            model_cfg = detect_model_config(model_path, algo)
            
            backtest_single_subagent(
                agent_name=agent_name,
                model_path=model_path,
                algo=algo,
                config=config,
                use_attention=model_cfg['use_attention'],
                attention_type=model_cfg['attention_type'],
                inferred_state_dim=model_cfg.get('state_dim'),
                model_type=model_type,  # ← 傳遞
            )
        else:
            # 預設：階層式 RL 回測
            print("[Backtest] 未指定回測模式，使用預設模式：階層式 RL 回測\n")
            backtest_hierarchical_agents(config)
        
        print(f"\n[Backtest] ✓ 回測完成！")
        print(f"  └─ 報告已保存到 ./backtest_reports/\n")
    
    except Exception as e:
        print(f"\n❌ [Backtest] 錯誤: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
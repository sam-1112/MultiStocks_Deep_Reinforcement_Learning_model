# MultiStocks_Deep_Reinforcement_Learning_model
論文實作：A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning

專案架構（Project Architecture）
``` 
.
├── configs
│   ├── agent
│   │   ├── final_cnn_attn.yaml
│   │   ├── mlp.yaml
│   │   ├── multi_cnn.yaml
│   │   └── timesnet.yaml
│   ├── algo
│   │   ├── a2c.yaml
│   │   ├── ddpg.yaml
│   │   ├── ddqn.yaml
│   │   └── ppo.yaml
│   ├── data
│   │   ├── ohlcv_fund.yaml
│   │   └── ohlcv.yaml
│   ├── defaults.yaml
│   └── env
│       ├── basic.yaml
│       └── with_fund.yaml
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   ├── backtest.py
│   └── train.py
├── tests
│   ├── test_agent_forward.py
│   ├── test_data_pipeline.py
│   └── test_env_step.py
└── trader
    ├── agents
    │   ├── base_agent.py
    │   ├── final_agent.py
    │   ├── mlp.py
    │   └── subagents
    │       ├── return_agent.py
    │       └── risk_agent.py
    ├── algos
    │   ├── a2c.py
    │   ├── base_algo.py
    │   ├── ddpg.py
    │   ├── ddqn.py
    │   └── ppo.py
    ├── data
    │   ├── datamodule.py
    │   ├── fundamentals.py
    │   └── indicators.py
    ├── envs
    │   ├── base_env.py
    │   └── trading_env.py
    ├── __init__.py
    ├── registry.py
    ├── trainer.py
    └── utils
        ├── logging.py
        ├── metrics.py
        └── seed.py
```


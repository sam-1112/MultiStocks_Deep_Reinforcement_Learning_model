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

# 基本面資料所需公式
1. Current Ratio = Total Current Assets / Total Current Liabilities
流動比率是用來表示一家公司利用現有資產 (present assets) 償還即期負債 (immediate liabilities) 的能力
2. Acid Test Ratio = (Cash & Cash Equivalents at Carrying Value + Cash And Short-Term Investments + Current Net Receivables) / Total Current Liabilities
速動比率衡量一家公司利用其現金及約當現金 (cash and cash equivalents) 來償還即期負債 (immediate liabilities) 的能力
3. Operating Cash Flow Ratio = Operating Cash Flow​ / Total Current Liabilities
營業現金流量比率是衡量短期償債能力的重要指標，表示公司以營運現金流量償還流動負債的能力
4. Debt Ratio = Total Liabilities​ / Total Assets
負債比率是衡量企業資產中有多少是由負債融資而來的重要槓桿指標
5. Debt to Equity Ratio = Total Liabilities / Common Stock + Retained Earnings
負債權益比率是評估公司資本結構槓桿程度的重要指標，表示每 1 元股東權益對應多少元負債
6. Interest Coverage Ratio = EBIT / Interest Expense
利息保障倍數是衡量公司以營運獲利償還利息能力的關鍵指標。它反映企業營運產生的收益能支付利息費用多少倍
7. Asset Turnover Ratio = Total Revenue / Average Total Assets
總資產週轉率是衡量公司利用資產產生營收效率的指標，顯示企業每 1 元資產能產生多少營收
8. Inventory Turnover Ratio = Cost of Goods and Services Sold / Average Inventory
存貨週轉率是衡量公司存貨管理效率的關鍵指標，表示企業一年內存貨被售出與補充的次數
9. Day Sales in Inventory Ratio = (inventory / Cost of Goods and Services Sold) x 365
存貨銷售天數是衡量公司庫存轉換為銷售的平均天數的重要指標
10. Return on Ratio = Net Income / Average Total Assets
報酬率類財務比率用來衡量公司相對於資產、權益或營收的獲利能力
11. Return on Equity Ratio = Net Income / Common Stock + Retained Earnings



./run_pipeline.sh view              # 查看所有 Agent 配置總覽
./run_pipeline.sh view-sub          # 查看 Sub-Agent 詳細配置
./run_pipeline.sh view-final        # 查看 Final Agent 詳細配置

# 修改演算法和模型
./run_pipeline.sh set-algo final ddpg mscnn
./run_pipeline.sh set-algo sub-0 a2c timesnet
./run_pipeline.sh set-algo sub-1 ddqn mlp

# 修改注意力機制
./run_pipeline.sh set-attention on sequence 4    # 啟用 (sequence 類型, 4 頭)
./run_pipeline.sh set-attention on simple 8      # 啟用 (simple 類型, 8 頭)
./run_pipeline.sh set-attention off              # 禁用

# 訓練單個 Agent
./run_pipeline.sh train final       # 訓練 Final Agent
./run_pipeline.sh train sub-0       # 訓練第 1 個 Sub-Agent
./run_pipeline.sh train all         # 訓練所有 (multi-agent 完整流程)

# 平行訓練 Sub-Agents (功能3)
./run_pipeline.sh train-parallel    # 使用預設 worker 數
./run_pipeline.sh train-parallel 4  # 使用 4 個 worker

./run_pipeline.sh eval final        # 評估 Final Agent
./run_pipeline.sh eval-all          # 評估所有 Agent

./run_pipeline.sh view                  # 查看配置
./run_pipeline.sh list-models           # 列出模型
./run_pipeline.sh clean-models          # 刪除模型
./run_pipeline.sh clean-logs            # 刪除日誌
./run_pipeline.sh status                # 檢查狀態
./run_pipeline.sh gpu-status            # 檢查 GPU

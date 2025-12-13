#!/bin/bash
# filepath: run_pipeline.sh
# 簡化版本：Multi-Stock DRL Trading Pipeline

# ============================================
# 設定
# ============================================
CONFIG_FILE="./configs/defaults.yaml"
MODEL_DIR="./models"
LOG_DIR="./logs"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ============================================
# 函數：顯示幫助
# ============================================

show_help() {
    cat << EOF
${BLUE}═══════════════════════════════════════════════════${NC}
${BLUE}  Multi-Stock DRL Trading Pipeline (簡化版)${NC}
${BLUE}═══════════════════════════════════════════════════${NC}

${GREEN}使用方式:${NC} ./run_pipeline.sh [命令] [選項]

${YELLOW}📋 查看配置:${NC}
  view                    查看所有 Agent 配置
  view-sub                查看 Sub-Agent 配置
  view-final              查看 Final Agent 配置

${YELLOW}🔧 修改配置:${NC}
  set-algo <agent> <algo> [model]
                          修改 Agent 使用的演算法和模型
                          例: set-algo sub-0 a2c mlp
                          例: set-algo final ddpg mscnn
  
  set-attention <on|off> [type] [heads]
                          啟用/禁用 Final Agent 注意力機制
                          例: set-attention on sequence 4
                          例: set-attention off

${YELLOW}🚀 訓練:${NC}
  train <agent>           訓練指定 Agent
                          agent: sub-0, sub-1, sub-2, final, all
                          例: train sub-0
                          例: train final
  
  train-parallel [N]      平行訓練 Sub-Agents
                          N: worker 數量 (預設: Sub-Agent 數量)
                          例: train-parallel 4
  
  train-final-with-subs   訓練 Final Agent (使用已訓練的 Sub-Agents)
                          前提: Sub-Agents 模型必須已存在
                          例: train-final-with-subs
  
  train-all               完整流程：平行訓練所有 Sub-Agents -> 訓練 Final Agent
                          例: train-all

${YELLOW}🧪 評估:${NC}
  eval <agent>            評估指定 Agent
                          例: eval final
  
  eval-all                評估所有 Agent

${YELLOW}📊 工具:${NC}
  list-models             列出已訓練的模型
  list-sub-models         列出已訓練的 Sub-Agent 模型
  clean-models            刪除所有模型檔案
  clean-logs              刪除所有日誌檔案
  status                  檢查系統狀態
  gpu-status              檢查 GPU 狀態

${YELLOW}ℹ️ 其他:${NC}
  help                    顯示此幫助

${GREEN}範例 - 完整流程 1（推薦）:${NC}
  # 平行訓練所有 Sub-Agents，然後訓練 Final Agent
  ./run_pipeline.sh train-all

${GREEN}範例 - 完整流程 2（逐步操作）:${NC}
  # 步驟 1: 查看配置
  ./run_pipeline.sh view
  
  # 步驟 2: 修改配置
  ./run_pipeline.sh set-algo final ddpg mscnn
  ./run_pipeline.sh set-attention on sequence 4
  
  # 步驟 3: 平行訓練 Sub-Agents (4 個 workers)
  ./run_pipeline.sh train-parallel 4
  
  # 步驟 4: 訓練 Final Agent (使用已訓練的 Sub-Agents)
  ./run_pipeline.sh train-final-with-subs
  
  # 步驟 5: 評估
  ./run_pipeline.sh eval-all

${GREEN}範例 - 逐個訓練:${NC}
  ./run_pipeline.sh train sub-0
  ./run_pipeline.sh train sub-1
  ./run_pipeline.sh train sub-2
  ./run_pipeline.sh train-final-with-subs

${GREEN}範例 - 修改配置:${NC}
  ./run_pipeline.sh set-algo sub-0 a2c timesnet
  ./run_pipeline.sh set-algo final ddpg mscnn
  ./run_pipeline.sh set-attention on sequence 4

EOF
}

# ============================================
# 函數：YAML 配置操作
# ============================================

# 讀取 YAML 值
get_yaml_value() {
    local file="$1"
    local path="$2"  # 如: "agent_mode.final_agent.algorithm"
    
    python3 << EOF
import yaml
with open('$file', 'r') as f:
    config = yaml.safe_load(f)

keys = '$path'.split('.')
value = config
for key in keys:
    value = value.get(key, {})
print(value if value else 'N/A')
EOF
}

# 設定 YAML 值
set_yaml_value() {
    local file="$1"
    local path="$2"  # 如: "agent_mode.final_agent.algorithm"
    local value="$3"
    
    python3 << EOF
import yaml
with open('$file', 'r') as f:
    config = yaml.safe_load(f)

keys = '$path'.split('.')
target = config
for key in keys[:-1]:
    if key not in target:
        target[key] = {}
    target = target[key]

target[keys[-1]] = $value

with open('$file', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print('✓ 已設定 $path = $value')
EOF
}

# ============================================
# 函數：查看配置
# ============================================

view_config() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Agent 配置總覽${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    python3 << 'EOF'
import yaml

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

agent_mode = config['agent_mode']['mode']
print(f"🔹 Agent 模式: {agent_mode.upper()}\n")

# Sub-Agents
print("📌 Sub-Agents:")
print("─" * 70)
print(f"{'索引':<6} {'名稱':<20} {'演算法':<10} {'模型':<15} {'注意力':<10}")
print("─" * 70)

sub_agents = config['agent_mode'].get('sub_agents', [])
for i, agent in enumerate(sub_agents):
    name = agent.get('name', 'Unknown')
    algo = agent.get('algorithm', 'N/A').upper()
    model = agent.get('model_type', 'N/A').upper()
    attn = "✗" if not agent.get('use_attention', False) else "✓"
    print(f"{i:<6} {name:<20} {algo:<10} {model:<15} {attn:<10}")

print("\n📌 Final Agent:")
print("─" * 70)

final = config['agent_mode'].get('final_agent', {})
algo = final.get('algorithm', 'N/A').upper()
model = final.get('model_type', 'N/A').upper()
use_attn = final.get('use_attention', False)
attn_type = final.get('attention_type', 'N/A') if use_attn else '─'
num_heads = final.get('num_heads', 'N/A') if use_attn else '─'

print(f"  演算法: {algo}")
print(f"  模型: {model}")
print(f"  注意力: {'✓ 啟用' if use_attn else '✗ 禁用'}")
if use_attn:
    print(f"    - 類型: {attn_type}")
    print(f"    - 頭數: {num_heads}")

print("\n" + "─" * 70)
EOF
}

view_sub_agents() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Sub-Agents 詳細配置${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    python3 << 'EOF'
import yaml

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

sub_agents = config['agent_mode'].get('sub_agents', [])

for i, agent in enumerate(sub_agents):
    print(f"[Sub-Agent {i}] {agent.get('name', 'Unknown')}")
    print(f"  Agent Type: {agent.get('agent_type', 'N/A')}")
    print(f"  Algorithm: {agent.get('algorithm', 'N/A').upper()}")
    print(f"  Model Type: {agent.get('model_type', 'N/A').upper()}")
    print(f"  Use Attention: {agent.get('use_attention', False)}")
    print()
EOF
}

view_final_agent() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Final Agent 詳細配置${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    python3 << 'EOF'
import yaml

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

final = config['agent_mode'].get('final_agent', {})

print(f"名稱: {final.get('name', 'Final Agent')}")
print(f"Agent Type: {final.get('agent_type', 'N/A')}")
print(f"Algorithm: {final.get('algorithm', 'N/A').upper()}")
print(f"Model Type: {final.get('model_type', 'N/A').upper()}")
print(f"\n注意力機制配置:")
print(f"  啟用: {'✓ 是' if final.get('use_attention', False) else '✗ 否'}")
if final.get('use_attention', False):
    print(f"  類型: {final.get('attention_type', 'N/A')}")
    print(f"  頭數: {final.get('num_heads', 4)}")
EOF
}

# ============================================
# 函數：修改配置
# ============================================

set_agent_algo() {
    local agent="$1"
    local algo="$2"
    local model="$3"
    
    if [ -z "$agent" ] || [ -z "$algo" ]; then
        echo -e "${RED}❌ 用法: set-algo <agent> <algo> [model]${NC}"
        echo "  agent: sub-0, sub-1, sub-2, final"
        echo "  algo: a2c, ddpg, ddqn, ppo"
        echo "  model: mlp, bilstm, lstm, mscnn, timesnet (可選)"
        return 1
    fi
    
    # 預設模型型別
    [ -z "$model" ] && model="mlp"
    
    echo -e "${YELLOW}修改 $agent 配置...${NC}"
    
    python3 << EOF
import yaml
import sys

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

agent = '$agent'.lower()
algo = '$algo'.lower()
model = '$model'.lower()

# 驗證輸入
valid_algos = ['a2c', 'ddpg', 'ddqn', 'ppo']
valid_models = ['mlp', 'bilstm', 'lstm', 'mscnn', 'timesnet']

if algo not in valid_algos:
    print(f"❌ 無效的演算法: {algo} (有效值: {', '.join(valid_algos)})")
    sys.exit(1)

if model not in valid_models:
    print(f"❌ 無效的模型: {model} (有效值: {', '.join(valid_models)})")
    sys.exit(1)

# 修改配置
if agent == 'final':
    config['agent_mode']['final_agent']['algorithm'] = algo
    config['agent_mode']['final_agent']['model_type'] = model
    print(f"✓ Final Agent 已更新:")
    print(f"  Algorithm: {algo.upper()}")
    print(f"  Model: {model.upper()}")
elif agent.startswith('sub-'):
    try:
        idx = int(agent.split('-')[1])
        sub_agents = config['agent_mode'].get('sub_agents', [])
        if 0 <= idx < len(sub_agents):
            sub_agents[idx]['algorithm'] = algo
            sub_agents[idx]['model_type'] = model
            print(f"✓ Sub-Agent {idx} 已更新:")
            print(f"  Name: {sub_agents[idx].get('name', 'Unknown')}")
            print(f"  Algorithm: {algo.upper()}")
            print(f"  Model: {model.upper()}")
        else:
            print(f"❌ Sub-Agent 索引超出範圍: {idx}")
            sys.exit(1)
    except ValueError:
        print(f"❌ 無效的 Agent 格式: {agent}")
        sys.exit(1)
else:
    print(f"❌ 無效的 Agent: {agent}")
    sys.exit(1)

with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF
}

set_attention() {
    local enable="$1"
    local attn_type="${2:-sequence}"
    local num_heads="${3:-4}"
    
    if [ -z "$enable" ]; then
        echo -e "${RED}❌ 用法: set-attention <on|off> [type] [heads]${NC}"
        echo "  type: simple, sequence (預設: sequence)"
        echo "  heads: 正整數 (預設: 4)"
        return 1
    fi
    
    if [ "$enable" != "on" ] && [ "$enable" != "off" ]; then
        echo -e "${RED}❌ 無效的選項: $enable (應為 on 或 off)${NC}"
        return 1
    fi
    
    local use_attn="True"
    [ "$enable" = "off" ] && use_attn="False"
    
    echo -e "${YELLOW}修改 Final Agent 注意力機制...${NC}"
    
    python3 << EOF
import yaml

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

if 'final_agent' not in config['agent_mode']:
    config['agent_mode']['final_agent'] = {}

config['agent_mode']['final_agent']['use_attention'] = $use_attn
if $use_attn:
    config['agent_mode']['final_agent']['attention_type'] = '$attn_type'
    config['agent_mode']['final_agent']['num_heads'] = $num_heads

with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

status = '✓ 已啟用' if $use_attn else '✓ 已禁用'
print(f"{status} Final Agent 注意力機制")
if $use_attn:
    print(f"  類型: $attn_type")
    print(f"  頭數: $num_heads")
EOF
}

# ============================================
# 函數：訓練
# ============================================

train_agent() {
    local agent="$1"
    
    if [ -z "$agent" ]; then
        echo -e "${RED}❌ 用法: train <agent>${NC}"
        echo "  agent: sub-0, sub-1, sub-2, final, all"
        return 1
    fi
    
    agent=$(echo "$agent" | tr '[:upper:]' '[:lower:]')
    
    # 設定操作模式
    python3 << EOF
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent_mode']['operation'] = 'training'
with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF
    
    echo -e "${MAGENTA}開始訓練: $agent${NC}\n"
    
    if [ "$agent" = "all" ]; then
        # 訓練所有 Agent（包括 sub 和 final）
        python3 main.py --train --config ./configs/defaults.yaml
    elif [ "$agent" = "final" ]; then
        # 訓練 Final Agent（假設 sub-agents 已訓練）
        python3 main.py --train --config ./configs/defaults.yaml
    elif [[ "$agent" =~ ^sub-[0-9]+$ ]]; then
        # 訓練單個 Sub-Agent
        idx=$(echo "$agent" | cut -d'-' -f2)
        python3 main.py --train --config ./configs/defaults.yaml --train-sub-agent "$idx"
    else
        echo -e "${RED}❌ 無效的 Agent: $agent${NC}"
        return 1
    fi
}

train_parallel() {
    local num_workers="$1"
    
    # 預設使用 Sub-Agent 數量
    if [ -z "$num_workers" ]; then
        num_workers=$(python3 << 'EOF'
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
sub_agents = config['agent_mode'].get('sub_agents', [])
print(len(sub_agents))
EOF
)
    fi
    
    echo -e "${MAGENTA}平行訓練 Sub-Agents (${num_workers} workers)${NC}\n"
    
    # 設定操作模式
    python3 << EOF
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent_mode']['operation'] = 'training'
config['agent_mode']['mode'] = 'multi-agent'
config['agent_mode']['num_workers'] = $num_workers
with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF
    
    python3 main.py --train --config ./configs/defaults.yaml --num-workers $num_workers
}

train_final_with_subs() {
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}訓練 Final Agent (使用已訓練的 Sub-Agents)${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}\n"
    
    # 檢查 Sub-Agent 模型是否存在
    python3 << 'PYEOF'
import os
import yaml

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

sub_agents = config['agent_mode'].get('sub_agents', [])
missing_models = []

print("檢查 Sub-Agent 模型...")
print("─" * 70)

for i, agent in enumerate(sub_agents):
    name = agent.get('name', f'Sub-Agent-{i}')
    model_path = f"./models/sub_agents/{name}_agent.pth"
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024 / 1024  # MB
        print(f"  ✓ [{i}] {name}: {size:.2f} MB")
    else:
        print(f"  ✗ [{i}] {name}: NOT FOUND")
        missing_models.append(name)

print("─" * 70)

if missing_models:
    print(f"\n❌ 缺少以下模型:")
    for model in missing_models:
        print(f"  - {model}")
    print(f"\n請先執行以下命令訓練 Sub-Agents:")
    print(f"  ./run_pipeline.sh train-parallel [N]  # 平行訓練所有 Sub-Agents")
    print(f"  或逐個訓練:")
    for i in range(len(sub_agents)):
        print(f"  ./run_pipeline.sh train sub-{i}")
    exit(1)
else:
    print(f"\n✓ 所有 Sub-Agent 模型已就緒\n")

PYEOF

    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # 設定操作模式
    python3 << EOF
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent_mode']['operation'] = 'training'
config['agent_mode']['mode'] = 'multi-agent'
config['agent_mode']['train_final_only'] = True
with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF

    echo -e "${MAGENTA}開始訓練 Final Agent...${NC}\n"
    python3 main.py --train --config ./configs/defaults.yaml --train-final-only
}

train_all() {
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}完整流程：平行訓練 Sub-Agents → 訓練 Final Agent${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}\n"
    
    # 獲取 Sub-Agent 數量
    num_workers=$(python3 << 'EOF'
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
sub_agents = config['agent_mode'].get('sub_agents', [])
print(len(sub_agents))
EOF
)
    
    echo -e "${MAGENTA}[步驟 1/2] 平行訓練 ${num_workers} 個 Sub-Agents...${NC}\n"
    train_parallel $num_workers
    
    echo -e "\n${MAGENTA}[步驟 2/2] 訓練 Final Agent...${NC}\n"
    train_final_with_subs
    
    echo -e "\n${GREEN}═══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ 完整流程完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}\n"
}

# ============================================
# 函數：評估
# ============================================

eval_agent() {
    local agent="$1"
    
    if [ -z "$agent" ]; then
        echo -e "${RED}❌ 用法: eval <agent>${NC}"
        echo "  agent: sub-0, sub-1, sub-2, final"
        return 1
    fi
    
    # 設定操作模式
    python3 << EOF
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent_mode']['operation'] = 'evaluation'
with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF
    
    echo -e "${MAGENTA}評估 Agent: $agent${NC}\n"
    python3 main.py --eval --config ./configs/defaults.yaml
}

eval_all() {
    echo -e "${MAGENTA}評估所有 Agent${NC}\n"
    
    # 設定操作模式
    python3 << EOF
import yaml
with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['agent_mode']['operation'] = 'evaluation'
with open('./configs/defaults.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
EOF
    
    python3 main.py --eval --config ./configs/defaults.yaml
}

# ============================================
# 函數：工具
# ============================================

list_models() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  已訓練的模型${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    if ls ${MODEL_DIR}/*.pth 1>/dev/null 2>&1; then
        echo -e "${YELLOW}📌 主模型 (Final Agent):${NC}"
        ls -lh ${MODEL_DIR}/*.pth 2>/dev/null | awk '{print "  📦 " $9 " (Size: " $5 ")"}'
    fi
    
    if ls ${MODEL_DIR}/sub_agents/*.pth 1>/dev/null 2>&1; then
        echo -e "\n${YELLOW}📌 Sub-Agent 模型${NC}"
        ls -lh ${MODEL_DIR}/sub_agents/*.pth 2>/dev/null | awk '{print "  📦 " $9 " (Size: " $5 ")"}'
    fi
    
    if ! ls ${MODEL_DIR}/*.pth ${MODEL_DIR}/sub_agents/*.pth 1>/dev/null 2>&1; then
        echo -e "${YELLOW}無已訓練的模型${NC}"
    fi
    echo ""
}

list_sub_models() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  已訓練的 Sub-Agent 模型${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    python3 << 'EOF'
import os
import yaml

with open('./configs/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

sub_agents = config['agent_mode'].get('sub_agents', [])

print(f"{'索引':<6} {'名稱':<25} {'演算法':<12} {'模型':<15} {'狀態':<10} {'大小':<10}")
print("─" * 90)

for i, agent in enumerate(sub_agents):
    name = agent.get('name', f'Sub-Agent-{i}')
    algo = agent.get('algorithm', 'N/A').upper()
    model_type = agent.get('model_type', 'N/A').upper()
    
    model_path = f"./models/sub_agents/{name}_agent.pth"
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024 / 1024  # MB
        status = "✓ 存在"
        print(f"{i:<6} {name:<25} {algo:<12} {model_type:<15} {status:<10} {size:.2f} MB")
    else:
        status = "✗ 缺失"
        print(f"{i:<6} {name:<25} {algo:<12} {model_type:<15} {status:<10} {'─':<10}")

print("─" * 90)
EOF
    echo ""
}

clean_models() {
    echo -e "${YELLOW}確定要刪除所有模型檔案嗎？(y/N)${NC}"
    read -r confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf ${MODEL_DIR}/*.pth ${MODEL_DIR}/sub_agents/*.pth 2>/dev/null
        echo -e "${GREEN}✓ 模型檔案已清理${NC}"
    fi
}

clean_logs() {
    echo -e "${YELLOW}確定要刪除所有日誌檔案嗎？(y/N)${NC}"
    read -r confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf ${LOG_DIR}/*.log ${LOG_DIR}/*.csv 2>/dev/null
        echo -e "${GREEN}✓ 日誌檔案已清理${NC}"
    fi
}

check_status() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  系統狀態${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    echo -e "${YELLOW}Python & PyTorch:${NC}"
    python3 --version 2>&1 | sed 's/^/  /'
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null
    python3 -c "import torch; print(f'  CUDA: {\"可用\" if torch.cuda.is_available() else \"不可用\"}')" 2>/dev/null
    
    echo -e "\n${YELLOW}目錄:${NC}"
    [ -d "./data" ] && echo "  ✓ data/" || echo "  ✗ data/"
    [ -d "$MODEL_DIR" ] && echo "  ✓ $MODEL_DIR/" || echo "  ✗ $MODEL_DIR/"
    [ -d "$LOG_DIR" ] && echo "  ✓ $LOG_DIR/" || echo "  ✗ $LOG_DIR/"
    
    echo -e "\n${YELLOW}訓練進程:${NC}"
    local count=$(pgrep -f "python.*main.py" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}● 有 $count 個訓練進程執行中${NC}"
    else
        echo "  ○ 無訓練進程"
    fi
    echo ""
}

check_gpu() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  GPU 狀態${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
    
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader | awk '{print "  GPU " $1 ": " $2 " (" $3 ", 使用: " $4 ", 利用率: " $5 ")"}'
    else
        python3 << 'EOF'
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
    fi
    echo ""
}

# ============================================
# 主程序
# ============================================

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    help|-h|--help)
        show_help
        ;;
    view)
        view_config
        ;;
    view-sub|view-sub-agents)
        view_sub_agents
        ;;
    view-final|view-final-agent)
        view_final_agent
        ;;
    set-algo)
        set_agent_algo "$2" "$3" "$4"
        ;;
    set-attention)
        set_attention "$2" "$3" "$4"
        ;;
    train)
        train_agent "$2"
        ;;
    train-parallel)
        train_parallel "$2"
        ;;
    train-final-with-subs)
        train_final_with_subs
        ;;
    train-all)
        train_all
        ;;
    eval)
        eval_agent "$2"
        ;;
    eval-all)
        eval_all
        ;;
    list-models)
        list_models
        ;;
    list-sub-models)
        list_sub_models
        ;;
    clean-models)
        clean_models
        ;;
    clean-logs)
        clean_logs
        ;;
    status)
        check_status
        ;;
    gpu-status)
        check_gpu
        ;;
    backtest|backtest-all)
        echo -e "${MAGENTA}回測所有模型${NC}\n"
        python3 backtest_cli.py --all --num-episodes 10
        ;;
    
    backtest-agent)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ 用法: backtest-agent <agent_name>${NC}"
            return 1
        fi
        echo -e "${MAGENTA}回測 Agent: $2${NC}\n"
        python3 backtest_cli.py --agent "$2" --num-episodes 10
        ;;
    
    backtest-final)
        echo -e "${MAGENTA}回測 Final Agent${NC}\n"
        python3 backtest_cli.py --agent "Final_Agent" --num-episodes 10
        ;;
    *)
        echo -e "${RED}❌ 未知命令: $1${NC}"
        echo "使用 ./run_pipeline.sh help 查看幫助"
        exit 1
        ;;
esac
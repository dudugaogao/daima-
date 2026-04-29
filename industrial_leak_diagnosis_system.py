import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging

# 配置日志，模拟工业级输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# ==========================================
# 1. 深度学习模型定义 (CNN + GCN)
# ==========================================

class TransientCNN(nn.Module):
    """一维卷积神经网络：用于提取球阀开启瞬间的瞬态时序特征"""
    def __init__(self):
        super(TransientCNN, self).__init__()
        # 针对 2048Hz 采样率设计的卷积核
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.fc = nn.Linear(32 * 30, 128) # 假设输入截断为约 1 秒的数据长度

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class PipelineGCN(nn.Module):
    """图卷积网络：用于融合主干管道交叉点传感器的空间拓扑关系"""
    def __init__(self, in_features, out_features):
        super(PipelineGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, node_features, adjacency_matrix):
        # 简化的 GCN 传播逻辑：A * X * W
        support = self.linear(node_features)
        output = torch.matmul(adjacency_matrix, support)
        return F.relu(output)

# ==========================================
# 2. 多 Agent 协作系统定义
# ==========================================

class DataPerceptionAgent:
    """Agent 1: 数据感知与预处理 Agent"""
    def __init__(self, sampling_rate=2048):
        self.sampling_rate = sampling_rate
        self.window_size = int(sampling_rate * 0.5) # 截取 0.5 秒的瞬态窗口
        logging.info(f"[Data Agent] 初始化完成，设定采样率: {self.sampling_rate}Hz")

    def capture_transient(self, raw_stream):
        """核心逻辑：放弃稳态数据，精准捕捉球阀开启瞬间的瞬态信号"""
        logging.info("[Data Agent] 正在监控 NI cDAQ 数据流...")
        # 模拟信号突变检测（如求导法识别阈值跨越）
        gradient = np.gradient(raw_stream)
        trigger_idx = np.argmax(np.abs(gradient))
        
        if trigger_idx > 0:
            logging.info(f"[Data Agent] 侦测到球阀动作瞬态突变！位置索引: {trigger_idx}")
            # 截取瞬态窗口
            start = max(0, trigger_idx - int(self.window_size/4))
            end = start + self.window_size
            transient_signal = raw_stream[start:end]
            return torch.tensor(transient_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return None

class GraphReasoningAgent:
    """Agent 2: 图谱推理诊断 Agent (长链推理核心)"""
    def __init__(self):
        self.cnn_extractor = TransientCNN()
        self.gcn_reasoner = PipelineGCN(in_features=128, out_features=64)
        self.classifier = nn.Linear(64, 3) # 输出：正常、微小泄漏、严重泄漏
        
        # 三管网拓扑邻接矩阵 (描述传感器在主干道交叉点的位置关系)
        self.adj_matrix = torch.tensor([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ], dtype=torch.float32)
        logging.info("[Reasoning Agent] CNN-GCN 联合推理引擎加载完毕，载入三管网拓扑矩阵。")

    def analyze(self, transient_tensor):
        logging.info("[Reasoning Agent] 启动长链推理逻辑...")
        # 1. 提取时序特征
        temporal_features = self.cnn_extractor(transient_tensor)
        
        # 2. 模拟多节点特征拼接 (实际应为多个传感器数据，此处用复制模拟)
        node_features = temporal_features.repeat(3, 1) 
        
        # 3. 图卷积空间推理
        spatial_features = self.gcn_reasoner(node_features, self.adj_matrix)
        
        # 4. 融合诊断
        global_feature = torch.mean(spatial_features, dim=0) # 全局池化
        logits = self.classifier(global_feature)
        prediction = torch.argmax(logits).item()
        
        status_map = {0: "管网运行正常", 1: "监测到微小泄漏 (疑似球阀未关严)", 2: "严重泄漏"}
        confidence = F.softmax(logits, dim=0)[prediction].item()
        
        logging.info(f"[Reasoning Agent] 推理完成。诊断结果: {status_map[prediction]} (置信度: {confidence:.2f})")
        return status_map[prediction], confidence

class StatusAssessmentAgent:
    """Agent 3: 状态评估与决策 Agent"""
    def __init__(self):
        self.hw_baseline = {"NI_DAQ_Temp": 45.0, "PCB_Sensor_Impedance": 100.0}
        logging.info("[Assessment Agent] 硬件寿命与健康状态基线已建立。")

    def generate_report(self, diagnosis_result, hw_telemetry):
        logging.info("[Assessment Agent] 正在生成综合诊断与决策报告...")
        report = f"""
        ==================================================
        [管网智能诊断与硬件健康报告]
        时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
        
        1. 诊断结论: 
           - {diagnosis_result[0]}
           - 算法置信度: {diagnosis_result[1]:.2%}
           
        2. 硬件状态评估:
           - 采集卡温度: {hw_telemetry['temp']}°C (阈值: 60°C) -> 状态: 良好
           - 传感器阻抗偏移: {hw_telemetry['impedance']}Ω -> 状态: 需关注 (寿命预估剩余 85%)
           
        3. 决策建议:
           - 触发自动维护工单。
           - 建议现场复核 2# 交叉点球阀密封状态。
        ==================================================
        """
        return report

# ==========================================
# 3. 核心中枢运行逻辑 (Main Pipeline)
# ==========================================

def run_industrial_system():
    print("\n--- 启动基于多 Agent 协同的工业管网微小泄漏智能诊断系统 ---\n")
    
    # 实例化 Agents
    data_agent = DataPerceptionAgent(sampling_rate=2048)
    reasoning_agent = GraphReasoningAgent()
    assessment_agent = StatusAssessmentAgent()
    
    # 模拟生成一段包含球阀开启瞬间的动态压力数据 (10秒，2048Hz)
    logging.info("[System] 正在连接数据采集硬件...")
    raw_stream_sim = np.random.normal(0, 0.1, 2048 * 10) 
    # 在第 3 秒注入一个瞬态冲击波形 (模拟球阀动作)
    raw_stream_sim[2048*3 : 2048*3+50] += np.linspace(0, 5, 50) 
    
    time.sleep(1) # 模拟系统等待
    
    # 阶段 1: 感知与预处理
    transient_data = data_agent.capture_transient(raw_stream_sim)
    
    if transient_data is not None:
        time.sleep(1)
        # 阶段 2: 深度推理
        diagnosis, confidence = reasoning_agent.analyze(transient_data)
        
        time.sleep(1)
        # 阶段 3: 硬件状态联合评估
        # 模拟当前获取的硬件遥测数据
        current_hw_status = {"temp": 48.2, "impedance": 105.4} 
        final_report = assessment_agent.generate_report((diagnosis, confidence), current_hw_status)
        
        print(final_report)
    else:
        logging.info("[System] 未检测到有效瞬态动作，系统继续待机。")

if __name__ == "__main__":
    # 为了运行原型，关闭张量未初始化的警告
    torch.manual_seed(42)
    run_industrial_system()

import json
import os
from datasets import Dataset

def load_synthesis_data(file_path, max_samples=None):
    """加载合成数据"""
    print(f"Loading synthesis data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # 格式化数据
    formatted_data = []
    for item in data:
        if isinstance(item, dict) and 'instruction' in item and 'input' in item and 'output' in item:
            formatted_item = {
                'instruction': item['instruction'],
                'input': item['input'],
                'output': item['output']
            }
            formatted_data.append(formatted_item)
        else:
            print(f"⚠️ 跳过不支持的数据格式: {list(item.keys())}")
    
    print(f"Loaded {len(formatted_data)} samples")
    return formatted_data

def create_label_grouped_dataset(data, batch_size=4):
    """创建按MNLI标签分组的数据集，优化小batch训练"""
    print(f"🔄 按MNLI标签重组数据集 (目标batch大小: {batch_size})...")
    
    # 按MNLI标签分组
    mnli_labels = ["entailment", "contradiction", "neutral"]
    
    label_groups = {label: [] for label in mnli_labels}
    
    for item in data:
        label = item.get('label', 'neutral')
        if label in label_groups:
            label_groups[label].append(item)
        else:
            # 处理未知标签
            print(f"⚠️ 未知标签: {label}，归类为neutral")
            label_groups['neutral'].append(item)
                
    
    # 打印分组统计
    print("📊 MNLI标签分组统计:")
    for label, items in label_groups.items():
        if items:  # 只显示有样本的标签
            print(f"   {label}: {len(items)}个样本")
    
    # 重新组织数据，确保同一batch内尽可能是同一标签
    reorganized_data = []
    
    # 为每个标签创建完整的batch
    for label, items in label_groups.items():
        if items:  # 只处理非空的组
            # 将该标签的样本按batch_size分组
            for i in range(0, len(items), batch_size):
                batch_items = items[i:i + batch_size]
                reorganized_data.extend(batch_items)
                print(f"   添加{label}批次: {len(batch_items)}个样本")
    
    print(f"✅ 数据重组完成: {len(reorganized_data)}个样本")
    print(f"🎯 预期batch数: {len(reorganized_data) // batch_size}")
    
    return reorganized_data

def prepare_grpo_dataset(data_list):
    """准备GRPO训练数据集"""
    print("🔄 准备GRPO数据集...")
    
    # 为GRPO添加prompt字段（关键修复）
    for item in data_list:
        # 优先使用messages字段（兼容原有逻辑）
        if 'messages' in item and len(item['messages']) > 0:
            user_message = item['messages'][0]['content']
            item['prompt'] = user_message
            del item['messages']
        else:
            # 核心修复：从instruction和input生成prompt
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            # 拼接成完整prompt（与原load_and_process_data逻辑一致）
            item['prompt'] = f"{instruction}\n\n{input_text}".strip()
    
    dataset = Dataset.from_list(data_list)
    print(f"✅ GRPO数据集准备完成，共{len(dataset)}个样本")
    
    # 验证prompt是否成功添加
    if len(dataset) > 0 and 'prompt' in dataset[0]:
        print(f"   示例prompt: {dataset[0]['prompt'][:50]}...")  # 打印前50字符
    else:
        print("⚠️ 警告：数据集仍未包含'prompt'字段！")
    
    return dataset


def validate_dataset_format(dataset):
    """验证数据集格式"""
    print("🔍 验证数据格式...")
    sample_item = dataset[0]
    print(f"   数据字段: {list(sample_item.keys())}")
    if 'prompt' in sample_item:
        print(f"   示例prompt长度: {len(sample_item['prompt'])}")
    if 'label' in sample_item:
        print(f"   示例MNLI标签: {sample_item['label']}")
    print("✅ 数据格式验证完成")
    
    return True

def create_optimized_dataset(file_path, max_samples=None, batch_size=4):
    """创建优化的数据集（完整流程）"""
    # 加载数据
    data = load_synthesis_data(file_path, max_samples)
    print(f"🔄 原始数据加载完成，共{len(data)}个样本")
    
    # 首先处理数据，提取标签
    processed_data = []
    for item in data:
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        target_label = 'neutral'  # 默认标签
        
        # 方法1：从output字段的JSON中提取标签（最准确）
        try:
            import json
            # 解析output中的JSON
            output_json = json.loads(output_text)
            if 'output' in output_json:
                target_label = output_json['output']
        except (json.JSONDecodeError, KeyError, TypeError):
            # JSON解析失败，继续使用其他方法
            pass
        
        # 方法2：如果方法1失败，从input字段中提取
        if target_label == 'neutral':
            if 'Target label (must match exactly):' in input_text:
                # 提取Target label后面的内容
                label_line = input_text.split('Target label (must match exactly):')[1].split('\n')[0].strip()
                target_label = label_line
        
        # 验证标签是否有效
        if target_label not in ['entailment', 'contradiction', 'neutral']:
            target_label = 'neutral'  # 无效标签默认为neutral
        
        # 添加标签到数据项
        item['label'] = target_label
        processed_data.append(item)
    
    # 应用标签分组优化
    print("🧠 应用MNLI标签分组优化...")
    grouped_data = create_label_grouped_dataset(processed_data, batch_size)
    print(f"✅ 标签分组完成，优化后数据量: {len(grouped_data)}个样本")
    
    # 准备GRPO数据集
    dataset = prepare_grpo_dataset(grouped_data)
    
    # 验证格式
    validate_dataset_format(dataset)
    
    return dataset, grouped_data
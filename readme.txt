##### und_sft #####
理解模型的监督微调，任务设定为物体定位
1. 数据集：目标检测数据集
2. 基座模型：qwen2.5 vl 3b instruct
3. 训练框架：accelerate微调
4. 训练完成后，使用und_infer进行推理，测试微调效果
预估显存占用量大于24G，因此使用A100 80G进行微调

关于EMA模型所需要的训练步数：
1. 最小训练步数
10,000步 - 这是EMA开始发挥稳定效果的最小步数
2. 推荐训练步数
50,000步 - 这是您的配置下获得良好EMA效果的最佳平衡点
3. 理想训练步数
100,000步 - 这是获得最佳EMA效果的步数

general_utils文件夹中有llm_generator和mllm_generator，是自己实现的llm解码函数。更加灵活可控。
但是注意llm_generator和mllm_generator需要进行修改才能放到自己的inference过程中去。
不要直接从general_utils中import函数


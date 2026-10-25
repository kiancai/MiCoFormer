# 呼吸道六配置50轮

基于dev28de0ee，继承已验收b7afbe6域适配基础设施。六组独立QKV普通/LoRA+、ratio32、LoRA+128、EVA固定rank初始化、LoRA+Norm。

源合同tmp/20261025_respiratory_peft_next50/PLAN.md；环境MCF_LONG_ROOT绑定独立任务根。先执行训练集EVA校准并封存初始化，再prepare并verify；runner pipeline每卡有限三个任务，各自门控后固定50轮，只有e50提取/评估。无安装、scancel或全参数训练。

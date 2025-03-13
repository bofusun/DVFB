# Unsupervised Zero-Shot Reinforcement Learning via Dual-Value Forward-Backward Representation

Code for "Unsupervised Zero-Shot Reinforcement Learning via Dual-Value Forward-Backward Representation", ICLR 2025 paper.

---

# Introduction

we propose a novel **D**ual **V**alue **F**orward-**B**ackward representation (**DVFB**) framework with a contrastive entropy intrinsic reward to achieve both zero-shot generalization and fine-tuning adaptation in online URL.
On the one hand, we demonstrate that poor exploration in forward-backward representations can lead to limited data diversity in online URL, impairing successor measures, and ultimately constraining generalization ability.
To address this issue, the DVFB framework learns successor measures through a skill value function while promoting data diversity through an exploration value function, thus enabling zero-shot generalization.
On the other hand, and somewhat surprisingly, by employing a straightforward dual-value fine-tuning scheme combined with a reward mapping technique, the pre-trained policy further enhances its performance through fine-tuning on downstream tasks, building on its zero-shot performance.
Through extensive multi-task generalization experiments, DVFB demonstrates both superior zero-shot generalization and fine-tuning adaptation abilities, surpassing state-of-the-art (SOTA) URL methods.

![1740414636720](https://github.com/user-attachments/assets/1f7d0c60-35fb-4eec-a108-ed9c0d711b1a)

---
# Quick Start

1. Setting up repo
```
git clone https://github.com/bofusun/DVFB
```
2. Install Dependencies
```
conda create -n DVFB python=3.8
conda activate DVFB
cd DVFB
pip install -r requirements.txt
```
3. Train

(1) Pretrain DVFB in quadruped domain
```
python my_pretrain_new.py agent=dvfb domain=quadruped obs_type=states seed=10
```
(2) Pretrain DVFB without contrative entropy reward in quadruped domain
```
python my_pretrain_new.py agent=dvfb0 domain=quadruped obs_type=states seed=10
```
(3) Fintune DVFB in quadruped walk task
```
python my_finetune_new_fb.py agent=dvfb task=quadruped_walk obs_type=states load_seed=10 seed=10
```
(4) Fintune DVFB without contrative entropy reward in quadruped walk task
```
python my_finetune_new_fb.py agent=dvfb0 task=quadruped_walk obs_type=states load_seed=10 seed=10
```




1. 配置虚拟窗口
apt-get install xvfb
Xvfb :1 -screen 0 1280x1024x16 &
export DISPLAY=:1
2. tmux中conda不一致问题
which conda显示不一样
export PATH=/root/deeplearningnew/anaconda3/bin:$PATH
source ~/.bashrc
CUDA_VISIBLE_DEVICES=4 python my_skillcombine.py --algorithm smm --domain walker  --task_name walk --obs_type states --seed 0
3. metaworld
apt-get install libosmesa6-dev
apt-get install libglew-dev
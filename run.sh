CUDA_VISIBLE_DEVICES=0 python  main.py  --model rsan  --save testrsan --ext sep --pre_train experiment/RSAN/model.pt --test_only --data_test Set5 --n_GPUs 1 --testscale 1.5+2.0+2.5+3.0+3.5+4.0
#CUDA_VISIBLE_DEVICES=4,5,6,7 python  main.py --model rsan --save rsan_0101 --ext sep   --lr_decay 200 --lr 1e-4 --epochs 800 --n_GPUs 4 --batch_size 16 --data_train DIV2K  --gamma 0.5




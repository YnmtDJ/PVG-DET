python ../main.py --dataset_root=../dataset \
                  --dataset_name=visdrone \
                  --baseline=fcos \
                  --backbone=pvt_v2_b2_li \
                  --num_classes=12 \
                  --min_size=800 \
                  --max_size=1333 \
                  --batch_size=2 \
                  --epochs=20 \
                  --warmup_epochs=2 \
                  --lr=0.0003 \
                  --device=cuda \
                  --checkpoint_path=../checkpoint/visdrone/fcos/pvt_v2_b2_li/checkpoint.pth \
                  --log_dir=../log/visdrone/fcos/pvt_v2_b2_li
python ../main.py --dataset_root=../dataset \
                  --dataset_name=COCO \
                  --baseline=fcos \
                  --backbone=pvt_v2_b2_li \
                  --num_classes=91 \
                  --min_size=400 \
                  --max_size=667 \
                  --batch_size=8 \
                  --epochs=12 \
                  --warmup_epochs=1 \
                  --lr=0.01 \
                  --device=cuda \
                  --checkpoint_path=../checkpoint/coco/fcos/pvt_v2_b2_li/checkpoint.pth \
                  --log_dir=../log/coco/fcos/pvt_v2_b2_li
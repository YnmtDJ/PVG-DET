python ../main.py --dataset_root=../dataset \
                  --dataset_name=VisDrone \
                  --baseline=FCOS \
                  --num_classes=12 \
                  --min_size=800 \
                  --max_size=1333 \
                  --batch_size=4 \
                  --epochs=18 \
                  --warmup_epochs=2 \
                  --lr=0.0006 \
                  --device=cuda \
                  --checkpoint_path=../checkpoint/visdrone/fcos/checkpoint.pth \
                  --log_dir=../log/visdrone/fcos/
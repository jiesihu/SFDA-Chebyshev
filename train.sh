python train_source.py --datasetS Domain_ISBI --data-dir /userhome/jiesi/dataset/Prostate_segmentation/SFDA

python adapt_to_target.py --target Domain_HK --data-dir /userhome/jiesi/dataset/Prostate_segmentation/SFDA --model-file /userhome/Code_SFDA/jiesi/SFDA-Cheby/logs/Domain_ISBI/20240131_164228.060498/checkpoint_197.pth.tar
python adapt_to_target.py --target Domain_BIDMC --data-dir /userhome/jiesi/dataset/Prostate_segmentation/SFDA --model-file /userhome/Code_SFDA/jiesi/SFDA-Cheby/logs/Domain_ISBI/20240131_164228.060498/checkpoint_197.pth.tar
python adapt_to_target.py --target Domain_I2CVB --data-dir /userhome/jiesi/dataset/Prostate_segmentation/SFDA --model-file /userhome/Code_SFDA/jiesi/SFDA-Cheby/logs/Domain_ISBI/20240131_164228.060498/checkpoint_197.pth.tar
python adapt_to_target.py --target Domain_UCL --data-dir /userhome/jiesi/dataset/Prostate_segmentation/SFDA --model-file /userhome/Code_SFDA/jiesi/SFDA-Cheby/logs/Domain_ISBI/20240131_164228.060498/checkpoint_197.pth.tar
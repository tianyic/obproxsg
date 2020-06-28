python obproxsg.py --backend mobilenetv1 --dataset_name cifar10 --lambda_ 0.0001 --max_epoch 200 --batch_size 128
python obproxsg.py --backend mobilenetv1 --dataset_name fashion_mnist --lambda_ 0.0001 --max_epoch 200 --batch_size 128
python obproxsg.py --backend resnet18 --dataset_name cifar10 --lambda_ 0.0001 --max_epoch 200 --batch_size 128
python obproxsg.py --backend resnet18 --dataset_name fashion_mnist --lambda_ 0.0001 --max_epoch 200 --batch_size 128

python obproxsg_plus.py --backend mobilenetv1 --dataset_name cifar10 --lambda_ 0.0001 --max_epoch 200 --batch_size 128
python obproxsg_plus.py --backend mobilenetv1 --dataset_name fashion_mnist --lambda_ 0.0001 --max_epoch 200 --batch_size 128
python obproxsg_plus.py --backend resnet18 --dataset_name cifar10 --lambda_ 0.0001 --max_epoch 200 --batch_size 128
python obproxsg_plus.py --backend resnet18 --dataset_name fashion_mnist --lambda_ 0.0001 --max_epoch 200 --batch_size 128

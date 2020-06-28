python run.py --backend mobilenetv1 --dataset_name cifar10 --optimizer obproxsg_plus --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1
python run.py --backend resnet18 --dataset_name cifar10 --optimizer obproxsg_plus --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1
python run.py --backend mobilenetv1 --dataset_name fashion_mnist --optimizer obproxsg_plus --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1
python run.py --backend resnet18 --dataset_name fashion_mnist --optimizer obproxsg_plus --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1

python run.py --backend mobilenetv1 --dataset_name cifar10 --optimizer obproxsg --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1
python run.py --backend resnet18 --dataset_name cifar10 --optimizer obproxsg --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1
python run.py --backend mobilenetv1 --dataset_name fashion_mnist --optimizer obproxsg --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1
python run.py --backend resnet18 --dataset_name fashion_mnist --optimizer obproxsg --lambda_ 0.0001 --max_epoch 200 --batch_size 128 -lr 0.1

python -m pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install pytorch-ignite pytest flake8 matplotlib tensorboard onnx ipython ipywidgets pandas optuna
python -m pip install cupy-cuda110
python -m pip install -e .

python -m pytest tests

python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1

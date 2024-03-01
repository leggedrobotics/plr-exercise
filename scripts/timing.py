from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from pytictac import Timer, CpuTimer
import time

from plr_exercise.models.cnn import Net

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(0)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)

    model = Net().to(device)

    with CpuTimer("cpu timer before warm up"):
        model(data)

    with Timer("gpu timer before warum up"):
        model(data)

    for i in range(100):
        model(data)

    with Timer("gpu timer after warm up"):
        model(data)

    with CpuTimer("cpu timer after warm up"):
        model(data)

    with Timer("100 x gpu timer after warm up"):
        for i in range(100):
            noise = torch.rand_like(data) * 0.0001
            model(data + noise)

    with CpuTimer("100 x cpu timer after warm up"):
        for i in range(100):
            noise = torch.rand_like(data) * 0.0001
            model(data + noise)

    with Timer("100 x gpu timer after warm up - with sync"):
        for i in range(100):
            noise = torch.rand_like(data) * 0.0001
            model(data + noise)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()

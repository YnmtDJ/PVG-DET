import torch


def func():
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name="./"),
    ) as p:
        x = []
        for i in range(100):
            x.append(torch.ones([1000000]).to('cuda'))
        return


if __name__ == '__main__':
    func()

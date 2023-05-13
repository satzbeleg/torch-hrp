import torch
import torch_hrp as thrp
from timeit import default_timer as timer


# HRP layer
model_hrp = thrp.HashedRandomProjection(
    output_size=1024,
    input_size=768,
    random_state=42
)


# requirements
# 2x GPUs with 80 Gb each (peak 68 Gb)
# CPU with 300 Gb RAM (peak 180 Gb)
# 175.793451 seconds
if __name__ == '__main__':
    x = torch.rand(int(20e6), 768)
    start = timer()
    pool = model_hrp.start_pool()
    hashed = model_hrp.infer(x, pool, chunk_size=int(45e5))
    model_hrp.stop_pool(pool)
    torch.cuda.empty_cache()
    print(f"{timer() - start: .6f} seconds")

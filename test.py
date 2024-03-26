"""def calculate_grid_size(numOfImg):
    if numOfImg == 1:
        return 1, 1
    elif numOfImg == 2:
        return 1, 2
    else:
        factors = []
        for i in range(1, int(numOfImg**0.5) + 1):
            if numOfImg % i == 0:
                factors.append((i, numOfImg // i))

        closest = min(factors, key=lambda f: abs(f[0] - f[1]))
        return closest


numOfImg = 10
row, col = calculate_grid_size(numOfImg)
print(row, "by", col)
"""

import torch

print("Torch version:", torch.__version__)

print("Is CUDA enabled?", torch.cuda.is_available())

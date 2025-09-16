import torch
import warp as wp
wp.init()
# パラメータ
param = torch.nn.Parameter(torch.tensor([2.0], device="cuda"))

optimizer = torch.optim.Adam([param], lr=0.1)

# Warp kernel (y = x^3)
@wp.kernel
def square_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] * x[tid]* x[tid]


@wp.kernel
def sum_kernel(arr: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(out, 0, arr[tid]*arr[tid])


for step in range(1000):
    optimizer.zero_grad()
    
    x_wp = wp.from_torch(param, dtype=wp.float32, requires_grad=True)
    y_wp = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)
    loss_wp = wp.zeros(1, dtype=wp.float32, device="cuda", requires_grad=True)

    # Warp Tape で勾配追跡
    tape = wp.Tape()
    with tape:
        wp.launch(square_kernel, dim=1, inputs=[x_wp, y_wp], device="cuda")
        wp.launch(sum_kernel, dim=1, inputs=[y_wp, loss_wp], device="cuda")

    # backward
    tape.backward(loss_wp)

    loss_val = wp.to_torch(loss_wp).item()
    if step % 10 == 0:
        print(f"step {step}: param = {param.item():.4f}, loss = {loss_val:.4f}, grad = {param.grad}")

    optimizer.step()

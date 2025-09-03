import torch  # type: ignore[import-not-found]


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_rows, dim = 1000, 2048
    w = torch.nn.Parameter(torch.randn((num_rows, dim), device=device))
    optimizer = torch.optim.SGD([w], lr=0.1)

    mask_upper = torch.triu(
        torch.ones((num_rows, num_rows), dtype=torch.bool, device=device),
        diagonal=1,
    )

    for step in range(1000):
        optimizer.zero_grad()
        normalized_w = torch.nn.functional.normalize(w, dim=1)
        sims = normalized_w @ normalized_w.T
        elements = sims[mask_upper]
        loss = elements.mean() + elements.std()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            w.copy_(torch.nn.functional.normalize(w, dim=1))
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}: loss={loss.item():.6f}")


if __name__ == "__main__":
    main()

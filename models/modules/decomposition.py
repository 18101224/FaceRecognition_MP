import torch
from torch import nn

__all__ = ['OrthogonalDecomposer']

class OrthogonalDecomposer(nn.Module):
    """
    입력 x ∈ R[bs, dim]을 원공간(dim)에서 두 성분(z1, z2)으로 분해.
    - z1, z2 모두 shape (bs, dim)
    - z1 ⟂ z2 (수치오차를 제외하면 직교), z1 + z2 = x
    - Cayley 변환으로 직교행렬 Q를 만든 뒤, Q의 앞 k(=dim//2) 개 열이 생성하는
      부분공간으로 사영하여 z1을 만들고, 나머지를 z2로 둠.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.k = dim // 2  # 내부에서 자동 결정 (필요 시 외부에서 속성으로 조정 가능)
        # 스큐-대칭 매개변수(학습 파라미터). 작은 초기값 권장.
        self.skew_params = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def _orthogonal_from_cayley(self, device, dtype):
        # A를 스큐-대칭으로 프로젝션
        A = torch.triu(self.skew_params, diagonal=1)
        A = A - A.T
        I = torch.eye(self.dim, device=device, dtype=dtype)
        # Q = (I + A)^{-1} (I - A)
        # (I + A)가 특이해지는 것을 막기 위해 작은 감쇠를 더할 수 있음(필요 시)
        Q = torch.linalg.solve(I + A, I - A)
        return Q

    def forward(self, x: torch.Tensor):
        """
        x: (bs, dim)
        returns:
          z1: (bs, dim)  # Q1부분공간 성분
          z2: (bs, dim)  # 보완 부분공간 성분
        """
        assert x.dim() == 2 and x.size(1) == self.dim, "x must be (bs, dim)"
        Q = self._orthogonal_from_cayley(x.device, x.dtype)        # (dim, dim)
        Q1 = Q[:, :self.k]                                         # (dim, k)
        # z1 = P1 x,  P1 = Q1 Q1^T
        z1 = (x @ Q1) @ Q1.T                                       # (bs, dim)
        # z2 = x - z1  (동치로 (x @ Q2) @ Q2^T 사용 가능)
        z2 = x - z1                                                # (bs, dim)
        return z1, z2

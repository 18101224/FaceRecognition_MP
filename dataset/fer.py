from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.extend('..')
from utils import init_quality_model, by_ml
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import os
import torch.distributed as dist
import pickle
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from .sampler import ImbalancedDatasetSampler
from .sampler_wrapper import DistributedSamplerWrapper
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import distributed as dist
from PIL import Image
from collections import deque
from concurrent.futures import ThreadPoolExecutor 
import random

__all__ = ['FER','FER_KFOLD','ClassBatchSampler']

classes = {
    'Angry' : 0,
    'Disgust' : 1,
    'Fear' : 2,
    'Happy' : 3,
    'Neutral' : 4,
    'Sad' : 5,
    'Surprise' : 6
}


class FER(Dataset):
    def __init__(self,args,transform,train=True, idx=True, balanced=False, imb_type=None, imb_factor:float=1.0, rand_number=566, crop_valid=False):
        '''
        AFfectNet : test 
        CAER : test
        RAF-DB : valid
        '''
        super().__init__()
        global classes
        self.root = args.dataset_path
        self.train = train
        post = 'train' if train else ('test' if ('Affect' in self.root or 'CAER' in self.root) else ('valid_balanced' if balanced else 'valid' ))
        offset = 1 if 'RAF' in self.root else 0 
        self.transform=transform
        self.paths = []
        self.labels = []
        if args.dataset_name == 'CAER':
            for key, value in classes.items():
                paths = sorted(glob(f'{self.root}/{post}/{key}/*'))
                self.paths += paths
                self.labels += [value]*len(paths)
        else:
            for i in range(int(args.num_classes)):
                paths = sorted(glob(f'{self.root}/{post}/{i+offset}/*'))
                self.paths += paths 
                self.labels += [i]*len(paths)
        self.labels = np.array(self.labels,dtype=np.int64)

        # Apply class imbalance (e.g., CIFAR-LT style) only for training
        if self.train and imb_type in ('exp', 'step') and imb_factor is not None and float(imb_factor) != 1.0:
            self._apply_imbalance(imb_type=imb_type, imb_factor=float(imb_factor), rand_number=int(rand_number))

        # Optionally crop validation/test split to minority class size for balanced evaluation
        if (not self.train) and crop_valid:
            self._crop_to_min_per_class(seed=int(rand_number))

        self.img_num_list = None 
        self.get_img_num_per_cls()
        self.idx = idx 
        # Optional in-memory preload of images for faster access
        self.pin_memory = getattr(args, 'pin_memory', False)
        self.path_to_index = {p: i for i, p in enumerate(self.paths)}
        self.preloaded_images = None
        if self.pin_memory:
            self.preloaded_images = []
            for p in tqdm(self.paths, disable=False):
                # Load image content into memory and detach from file handle
                img = Image.open(p)
                img = img.copy()
                self.preloaded_images.append(img)

    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        if self.preloaded_images is not None:
            img = self.preloaded_images[idx]
        else:
            img = Image.open(self.paths[idx])
        if isinstance(self.transform, list):
            samples = [tr(img) for tr in self.transform]
            if self.idx : 
                return samples, self.labels[idx] , idx
            else : 
                return samples, self.labels[idx] # len transforms, img 
        else:
            if self.idx : 
                return self.transform(img), self.labels[idx] , idx
            else : 
                return self.transform(img), self.labels[idx]
    
    def get_img_num_per_cls(self):
        if self.img_num_list is None:
            counter = Counter(self.labels)
            self.img_num_list = [0]*(np.max(self.labels)+1)
            for key, value in counter.items():
                self.img_num_list[key] = value
        return np.array(self.img_num_list)

    def _apply_imbalance(self, imb_type: str, imb_factor: float, rand_number: int = 0):
        """
        Subsample the current training set to follow a long-tailed distribution.

        - imb_type: 'exp' (exponential) or 'step' (half head, half tail like CIFAR-LT helper)
        - imb_factor: ratio of tail class to head class (e.g., 0.01)
        - rand_number: random seed for reproducibility
        """
        assert imb_type in ('exp', 'step')
        if imb_factor <= 0 or imb_factor > 1:
            # Guardrails; 1.0 means no change, (<0 or >1 not valid)
            return

        np.random.seed(rand_number)

        labels_np = np.asarray(self.labels, dtype=np.int64)
        paths_np = np.asarray(self.paths, dtype=object)
        classes = np.unique(labels_np)
        cls_num = int(classes.max()) + 1

        # Compute target per-class counts following CIFAR-LT logic
        # Use the maximum available per-class count as head count baseline
        per_class_counts = np.array([(labels_np == c).sum() for c in range(cls_num)], dtype=np.int64)
        if per_class_counts.sum() == 0:
            return
        img_max = float(per_class_counts.max())

        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for _ in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for _ in range(cls_num - cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))

        # Optional rotation of class order to avoid always making class 0 the head (mimic CIFAR helper)
        ordered_classes = np.concatenate([classes[1:], classes[:1]], axis=0)

        new_paths = []
        new_labels = []
        for the_class, the_img_num in zip(ordered_classes, img_num_per_cls):
            idx = np.where(labels_np == the_class)[0]
            if len(idx) == 0:
                continue
            np.random.shuffle(idx)
            take = min(int(the_img_num), len(idx))
            sel = idx[:take]
            new_paths.append(paths_np[sel])
            new_labels.append(np.full(take, int(the_class), dtype=np.int64))

        if len(new_paths) == 0:
            return

        self.paths = np.concatenate(new_paths).tolist()
        self.labels = np.concatenate(new_labels)
        # Reset helper caches derived from paths
        self.img_num_list = None
        # Rebuild path_to_index later in __init__

    def _crop_to_min_per_class(self, seed: int = 0):
        """
        For validation/test splits: crop each class to the size of the least frequent class.
        Selection is randomized with a deterministic seed for reproducibility.
        """
        if len(self.labels) == 0:
            return

        rng = np.random.RandomState(seed)
        labels_np = np.asarray(self.labels, dtype=np.int64)
        paths_np = np.asarray(self.paths, dtype=object)

        classes = np.unique(labels_np)
        # Determine minority class count among present classes
        per_class_counts = {c: int((labels_np == c).sum()) for c in classes}
        if len(per_class_counts) == 0:
            return
        min_count = int(min(per_class_counts.values()))
        if min_count <= 0:
            return

        kept_paths = []
        kept_labels = []
        for c in classes:
            idx = np.where(labels_np == c)[0]
            if len(idx) <= min_count:
                chosen = idx
            else:
                chosen = rng.choice(idx, size=min_count, replace=False)
            kept_paths.append(paths_np[chosen])
            kept_labels.append(np.full(len(chosen), int(c), dtype=np.int64))

        self.paths = np.concatenate(kept_paths).tolist()
        self.labels = np.concatenate(kept_labels)
        # Reset caches; indices and counts will be rebuilt downstream in __init__
        self.img_num_list = None

class FER_KFOLD(FER):
    def __init__(self, args, transform ,n_folds=5, fold_idx=0, train=True, random_seed=42):
        super().__init__(args, transform, train=True)  # Always load all data, split later

        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.random_seed = random_seed
        self.train = train
        self.labels = np.array(self.labels)
        
        # self.paths: numpy array of all image paths, shape = (num_samples,)
        self.paths = np.array(self.paths)
        # self.original_paths: deep copy of all image paths, shape = (num_samples,)
        self.original_paths = deepcopy(self.paths)
        # self.indices: list of (train_idx, val_idx) tuples for each fold, each idx is a numpy array of indices
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.indices = list(kf.split(X=self.paths, y=self.labels))
        # train_idx: indices for training samples in this fold, shape = (num_train_samples,)
        # val_idx: indices for validation samples in this fold, shape = (num_val_samples,)
        train_idx, val_idx = self.indices[fold_idx]
        
        self.val_idx = val_idx
        if train:
            selected_idx = train_idx
        else:
            selected_idx = val_idx
        self.paths = self.paths[selected_idx]
        self.labels = self.labels[selected_idx]
        # Convert back to torch tensor for labels
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.paths)


    def get_indice_db(self):
        result = torch.cat((torch.tensor(self.val_idx),torch.tensor([self.fold_idx]*len(self.val_idx)))) #deterministic. 
        return result

class FER_uni(FER):
    def __init__(self,args, transform, train, idx, c):
        super().__init__(args, transform, train, idx)
        indices = self.labels[self.labels == c]
        self.paths = self.paths[indices.tolist()]
        self.labels = self.labels[indices]
    
    def __len__(self):
        return len(self.labels)


class ClassBatchSampler(FER):
    def __init__(self, args, transform, train=True, idx=True, balanced=False,
                 seed=42, num_workers=0, image_size=(112,112),
                 replace_if_insufficient=True):
        super().__init__(args, transform, train=train, idx=idx, balanced=balanced)  # FER 초기화 [web:56]
        # labels, class_to_idx 설정 (기존 스타일 유지)
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # FER가 제공한 self.labels 사용 [web:56]
        num_classes = int(self.labels.max().item()) + 1 if len(self.labels) > 0 else 0  # 클래스 수 계산 [web:56]
        self.class_to_idx = [[] for _ in range(num_classes)]  # 클래스별 인덱스 버킷 [web:56]
        for i, c in enumerate(self.labels.tolist()):  # 각 샘플을 해당 클래스 리스트에 추가 [web:56]
            self.class_to_idx[c].append(i)  # 클래스 버킷 채우기 [web:56]

        # DDP 환경 자동 감지
        if dist.is_available() and dist.is_initialized():  # torch.distributed 초기화 확인 [web:66]
            self.world_size = dist.get_world_size()  # 월드 크기 [web:66]
            self.rank = dist.get_rank()  # 현재 랭크 [web:66]
        else:
            self.world_size = 1  # 단일 프로세스 기본값 [web:55]
            self.rank = 0  # 단일 프로세스 기본값 [web:55]

        # 상태
        self.seed = int(seed)  # 시드 저장 [web:55]
        self.epoch = 0  # 에폭 상태 [web:55]
        self.rng = random.Random(self._seed_for_epoch(self.epoch))  # 에폭 기반 RNG [web:55]
        self.num_workers = int(num_workers)  # 로딩 스레드 수 [web:56]
        self.image_size = image_size  # 기대 이미지 크기 [web:56]
        self.replace_if_insufficient = bool(replace_if_insufficient)  # tail 클래스 리필 정책 [web:46]

        # 전처리/로드 관련 (FER가 제공하는 프리로드 최적화 경로 사용)
        self._has_preloaded = getattr(self, 'preloaded_images', None) is not None  # 프리로드 여부 [web:56]
        self._path_to_index = getattr(self, 'path_to_index', None)  # 경로→인덱스 맵 [web:56]

        # 클래스별 deque 구성 (전역 무복원 pop 상태)
        self._build_deques()  # 클래스별 셔플 deque 초기화 [web:46]

    def _seed_for_epoch(self, epoch):
        return (self.seed + epoch * 1000003) & 0x7FFFFFFF  # 에폭별 결정적 시드 생성 [web:55]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)  # 에폭 갱신 [web:55]
        self.rng = random.Random(self._seed_for_epoch(self.epoch))  # RNG 재설정 [web:55]
        self._build_deques()  # 에폭 경계에서 셔플 리셋 [web:55]

    def _build_deques(self):
        self.deques = []  # 클래스별 남은 인덱스 저장 deque [web:46]
        for _, idxs in enumerate(self.class_to_idx):  # 각 클래스 버킷 순회 [web:46]
            if len(idxs) == 0:
                self.deques.append(deque())  # 빈 클래스 처리 [web:46]
            else:
                shuffled = list(idxs)  # 복사 [web:46]
                self.rng.shuffle(shuffled)  # 클래스 내부 셔플 [web:55]
                self.deques.append(deque(shuffled))  # deque 저장 [web:46]

    def _refill_if_needed(self, c: int):
        idxs = self.class_to_idx[c]  # 원본 버킷 [web:46]
        if len(idxs) == 0:
            return  # 빈 클래스면 리필 불가 [web:46]
        shuffled = list(idxs)  # 복사 [web:46]
        self.rng.shuffle(shuffled)  # 리필용 셔플 [web:55]
        self.deques[c].extend(shuffled)  # 뒤에 이어붙여 리필 [web:46]

    def _open(self, path):
        if self._has_preloaded and self._path_to_index is not None:  # 프리로드 경로 우선 사용 [web:56]
            i = self._path_to_index[path]  # 인덱스 조회 [web:56]
            img = self.preloaded_images[i]  # 프리로드 이미지 반환 [web:56]
            return img  # transform 단계에서 tensor/PIL 처리 [web:56]
        img = Image.open(path)  # 디스크에서 PIL Image 로딩 (lazy) [web:56]
        return img  # PIL 이미지 반환 [web:56]

    def _apply_transform(self, img):
        return self.transform(img) if self.transform is not None else img  # init에서 받은 transform 적용 [web:56]

    def _load_one(self, path):
        img = self._open(path)  # 경로 로딩 [web:56]
        img = self._apply_transform(img)  # 변환 적용 [web:56]
        return img  # 텐서 또는 PIL 반환(여기서는 텐서 기대) [web:56]

    def __len__(self):
        return len(self.labels)  # 전체 샘플 수 [web:56]

    # 전역 무복원 샘플링(현재 프로세스 컨텍스트 내 범위)
    def _sample_pairs_global(self, labels_batch: torch.Tensor, k: int):
        pairs = []  # (idx, class) 쌍 리스트 [web:46]
        for c in labels_batch.tolist():  # 배치의 각 라벨에 대해 [web:46]
            c = int(c)  # 클래스 정수 [web:46]
            dq = self.deques[c]  # 해당 클래스 deque [web:46]
            needed = k  # 필요한 개수 [web:46]
            while needed > 0:  # 필요한 만큼 pop [web:46]
                if len(dq) == 0:  # 고갈 시 [web:46]
                    if self.replace_if_insufficient:  # 리필 허용이면 리필 [web:46]
                        self._refill_if_needed(c)  # 클래스 리필 [web:46]
                        dq = self.deques[c]  # 참조 갱신 [web:46]
                        if len(dq) == 0:  # 여전히 비면 에러 [web:46]
                            raise ValueError(f"Class {c} empty; cannot refill.")  # 에러 처리 [web:46]
                    else:
                        raise ValueError(f"Class {c} insufficient for k={k}.")  # 엄격 모드 에러 [web:46]
                pairs.append((dq.popleft(), c))  # 무복원 pop [web:46]
                needed -= 1  # 카운트 감소 [web:46]
        return pairs  # 길이 = bs*k [web:46]

    # 단일 프로세스/비-DDP 경로
    def sample_pairs(self, labels_batch: torch.Tensor, k: int):
        """
        단일 프로세스 또는 외부 DDP 동기화 없이 사용하는 경로(비-DDP 권장):
        return: images (bs, k, 3, 112, 112), labels (bs, k) [web:56]
        """
        pairs = self._sample_pairs_global(labels_batch, k)  # 전역 무복원 pop [web:46]
        idxs = [i for i, _ in pairs]  # 인덱스 추출 [web:56]
        paths = [self.paths[i] for i in idxs]  # 경로 매핑 [web:56]

        if self.num_workers > 0:  # 스레드 병렬 로딩 [web:56]
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:  # ThreadPoolExecutor 사용 [web:56]
                imgs = list(ex.map(self._load_one, paths))  # 병렬 변환 [web:56]
        else:
            imgs = [self._load_one(p) for p in paths]  # 순차 로딩 [web:56]

        if not all(isinstance(t, torch.Tensor) for t in imgs):  # 변환 결과 검증 [web:56]
            raise TypeError("Transform must return torch.Tensor.")  # 타입 요구 [web:56]
        batch = torch.stack(imgs, dim=0)  # (bs*k, C, H, W) 스택 [web:56]
        C, H, W = batch.shape[1:]  # 형상 확인 [web:56]
        if not (C == 3 and H == 112 and W == 112):  # 크기 검증 [web:56]
            raise ValueError(f"Expected (3,112,112), got {(C,H,W)}.")  # 오류 보고 [web:56]

        y = torch.tensor([c for _, c in pairs], device=labels_batch.device, dtype=torch.long)  # 라벨 텐서 [web:56]
        total = batch.shape[0]  # 총 샘플 수 [web:56]
        if total % k != 0:  # 나머지 버리기 [web:56]
            drop = total % k  # 드롭 수 [web:56]
            if drop > 0:
                batch = batch[:-drop]  # 컷오프 [web:56]
                y = y[:-drop]  # 라벨 동기 컷오프 [web:56]
                total = batch.shape[0]  # 갱신 [web:56]
        bs = total // k  # 배치 크기 [web:56]
        batch = batch.view(bs, k, C, H, W)  # (bs, k, C, H, W) 리쉐이프 [web:56]
        y = y.view(bs, k)  # 라벨 리쉐이프 [web:56]
        return batch.to(labels_batch.device)  # 디바이스 정렬 [web:56]

    # --------- DDP 동기화 경로(전역 무복원 유지, ddp_shard=False 사용 의도) ---------

    def _all_gather_labels_object(self, labels_batch: torch.Tensor):
        """
        모든 랭크의 가변 길이 라벨을 객체 통신으로 수집(패딩/마스크 불필요).
        returns: labels_list_per_rank: List[List[int]] 길이 = world_size [web:131]
        """
        assert dist.is_available() and dist.is_initialized(), "DDP must be initialized."  # 전제 확인 [web:66]
        world_size = dist.get_world_size()  # 월드 크기 [web:66]
        local_labels = labels_batch.to(torch.long).tolist()  # 로컬 라벨 리스트화 [web:131]
        gather_list = [None for _ in range(world_size)]  # 수신 컨테이너 [web:131]
        dist.all_gather_object(gather_list, local_labels)  # 가변 길이 객체 수집 [web:131]
        return gather_list  # rank 순서대로 라벨 묶음 [web:131]

    def _make_shards_on_rank0(self, labels_list_per_rank, k: int):
        """
        rank0에서만 호출: 수집된 각 랭크 라벨 묶음 순서대로 전역 무복원 pop 수행 후
        해당 랭크에 필요한 pairs만큼 할당하여 shards를 만든다. [web:55]
        """
        shards = [[] for _ in range(len(labels_list_per_rank))]  # 랭크별 쿼터 [web:55]
        for r, labels_r in enumerate(labels_list_per_rank):  # 랭크 순회 [web:55]
            if len(labels_r) == 0:
                continue  # 빈 랭크 스킵 [web:55]
            pairs_r = self._sample_pairs_global(torch.as_tensor(labels_r, dtype=torch.long), k)  # 전역 pop [web:55]
            shards[r] = pairs_r  # 할당 [web:55]
        return shards  # world_size 길이의 shard 리스트 [web:55]

    def _broadcast_shards(self, shards):
        """
        rank0에서 만든 shards를 모든 랭크에 브로드캐스트한다(파이썬 객체 통신).
        각 랭크는 동일한 길이의 리스트 컨테이너를 보유해야 한다. [web:66]
        """
        assert dist.is_available() and dist.is_initialized(), "DDP must be initialized."  # 전제 [web:66]
        rank = dist.get_rank()  # 현재 랭크 [web:66]
        world_size = dist.get_world_size()  # 월드 크기 [web:66]
        if rank != 0:
            shards = [None for _ in range(world_size)]  # 수신 컨테이너 준비 [web:66]
        dist.broadcast_object_list(shards, src=0)  # rank0→전체 브로드캐스트 [web:66]
        return shards  # 모든 랭크 동일한 shards 수신 [web:66]


    def sample_pairs_dist_gathered(self, labels_batch: torch.Tensor, k: int):
        """
        DDP 안전 전역 무복원 샘플링 경로(요청한 대로 ddp_shard=False 유지):
        - 각 랭크: 로컬 labels_batch 가변 길이 입력 [web:131]
        - all_gather_object로 모든 랭크 라벨 수집 [web:131]
        - rank0: 전역 무복원 pop 및 랭크별 shard 생성 [web:55]
        - broadcast_object_list로 shards 전파 [web:66]
        - 각 랭크: shards[rank]만 소비하여 (bs_local, k, 3, 112, 112) 반환 [web:56]
        """
        if not (dist.is_available() and dist.is_initialized()):  # 비-DDP이면 일반 경로 [web:55]
            return self.sample_pairs(labels_batch, k)  # 단일 프로세스 반환 [web:56]

        # 1) 모든 랭크 라벨 수집(가변 길이 안전)
        labels_all = self._all_gather_labels_object(labels_batch)  # 리스트 수집 [web:131]

        # 2) rank0에서 shard 생성
        rank = dist.get_rank()  # 현재 랭크 [web:66]
        if rank == 0:
            shards = self._make_shards_on_rank0(labels_all, k)  # 전역 무복원 pop 및 분배 [web:55]
        else:
            shards = None  # 수신자 초기화 [web:66]

        # 3) shards 브로드캐스트
        shards = self._broadcast_shards(shards)  # rank0→전체 전파 [web:66]

        # 4) 로컬 shard 로딩 및 텐서화
        pairs_local = shards[rank]  # 내 shard 선택 [web:66]
        idxs = [i for i, _ in pairs_local]  # 인덱스 추출 [web:56]
        paths = [self.paths[i] for i in idxs]  # 경로 매핑 [web:56]

        if self.num_workers > 0:  # 스레드 로딩 [web:56]
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:  # 실행기 생성 [web:56]
                imgs = list(ex.map(self._load_one, paths))  # 병렬 적용 [web:56]
        else:
            imgs = [self._load_one(p) for p in paths]  # 순차 적용 [web:56]

        if not all(isinstance(t, torch.Tensor) for t in imgs):  # 타입 검증 [web:56]
            raise TypeError("Transform must return torch.Tensor.")  # 변환 파이프라인 요구 [web:56]

        batch = torch.stack(imgs, dim=0)  # (bs_local*k, C, H, W) [web:56]
        C, H, W = batch.shape[1:]  # 형상 추출 [web:56]
        if not (C == 3 and H == 112 and W == 112):  # 크기 검증 [web:56]
            raise ValueError(f"Expected (3,112,112), got {(C,H,W)}.")  # 오류 [web:56]

        y = torch.tensor([c for _, c in pairs_local], device=labels_batch.device, dtype=torch.long)  # 라벨 텐서 [web:56]

        total = batch.shape[0]  # 총 수 [web:56]
        if total % k != 0:  # 배치 정합성 처리 [web:56]
            drop = total % k  # 드롭할 수 [web:56]
            if drop > 0:
                batch = batch[:-drop]  # 트리밍 [web:56]
                y = y[:-drop]  # 동기 트리밍 [web:56]
                total = batch.shape[0]  # 갱신 [web:56]
        bs_local = total // k  # 로컬 배치 크기 [web:56]
        batch = batch.view(bs_local, k, C, H, W)  # (bs_local, k, C, H, W) [web:56]
        y = y.view(bs_local, k)  # 라벨 리쉐이프 [web:56]
        return batch.to(labels_batch.device).detach().clone() # 디바이스 정렬 [web:56]
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os, json, re
from glob import glob
from collections import Counter
from tqdm import tqdm 

__all__ = ['get_cifar_dataset', 'Large_dataset', 'CINIC10', 'CORe50', 'SmallNORB']

def get_cifar_dataset(dataset_name:str, root:str, imb_type:str, imb_factor:float, rand_number=0, train=True, transform=None, target_transform=None, download=True,):
    class CIFAR(torchvision.datasets.CIFAR10):
        def __init__(self, root, train, transform=None, target_transform=None, download=True, imb_type=None, imb_factor=None, rand_number=None):
            if dataset_name == 'cifar10':
                self.cls_num=10
            else:
                self.base_folder = 'cifar-100-python'
                self.url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
                self.filename = "cifar-100-python.tar.gz"
                self.tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
                self.train_list = [
                    ['train', '16019d7e3df5f24257cddd939b257f8d'],
                ]

                self.test_list = [
                    ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
                ]
                self.meta = {
                    'filename': 'meta',
                    'key': 'fine_label_names',
                    'md5': '7973b15100ade9c7d40fb424638fde48',
                }
                self.cls_num = 100
            super(CIFAR, self).__init__(root, train, transform, target_transform, download)
            np.random.seed(rand_number)
            if not train : 
                imb_factor = 1
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.labels = self.targets
            self.img_num_list = self.get_cls_num_list()

        def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
            img_max = len(self.data)/cls_num
            img_num_per_lcs = []
            if imb_type == 'exp':
                for cls_idx in range(cls_num):
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_lcs.append(int(num))
            elif imb_type == 'step':
                for cls_idx in range(cls_num // 2):
                    img_num_per_lcs.append(int(img_max))
                for cls_idx in range(cls_num // 2):
                    img_num_per_lcs.append(int(img_max * imb_factor))
            else:
                img_num_per_lcs.extend([int(img_max)] * cls_num)
            return img_num_per_lcs
        
        def gen_imbalanced_data(self, img_num_per_cls): 
            new_data = []
            new_targets = []
            targets_np = np.array(self.targets, dtype=np.int64)
            classes = np.unique(targets_np)
            classes = np.concatenate([classes[1:], classes[:1]], axis=0)
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.append(self.data[selec_idx, ...])
                new_targets.extend([the_class, ] * the_img_num)
            new_data = np.vstack(new_data)
            self.data = new_data
            self.targets = new_targets
        
        def get_cls_num_list(self):
            cls_num_list = []
            for i in range(self.cls_num):
                cls_num_list.append(self.num_per_cls_dict[i])
            return cls_num_list
        
        def __getitem__(self,idx):
            img, target = self.data[idx], int(self.targets[idx])
            img = Image.fromarray(img)
            
            if self.transform is not None : 
                if isinstance(self.transform, list):
                    samples = [tr(img) for tr in self.transform]
                else:
                    samples = self.transform(img)
            
            if self.target_transform is not None : 
                target = self.target_transform(target)

            return samples, target
        
        def __len__(self):
            return len(self.data)
        
    return CIFAR(root=root, train=train, transform=transform, target_transform=target_transform, download=download, imb_type=imb_type, imb_factor=imb_factor, rand_number=rand_number)


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _is_image_file(p: str) -> bool:
    return p.lower().endswith(_IMG_EXTS)


def _auto_base_dir(root: str, candidate_names=()):
    """
    If root contains a single child directory or known candidate directory,
    return that as base. Otherwise return root.
    """
    root = os.path.abspath(os.path.expanduser(str(root)))
    
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset_path not found: {root}")

    # Prefer known candidate subdirs if exist
    for name in candidate_names:
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            return cand

    # If it contains exactly one directory and no obvious split dirs, descend
    children = [os.path.join(root, x) for x in os.listdir(root)]
    dirs = [d for d in children if os.path.isdir(d)]
    files = [f for f in children if os.path.isfile(f)]
    if len(dirs) == 1 and len(files) == 0:
        return dirs[0]
    return root


def _limit_debug(paths, labels, debug: bool, n=2000, seed=42):
    if not debug:
        return paths, labels
    rng = np.random.RandomState(seed)
    n = min(n, len(labels))
    idx = rng.choice(np.arange(len(labels)), size=n, replace=False)
    idx = np.sort(idx)
    paths = [paths[i] for i in idx]
    labels = labels[idx]
    return paths, labels


class CINIC10(Dataset):
    """
    Expected folder (after extracting CINIC-10.tar.gz):
      <root>/
        train/<class_name>/*.png
        valid/<class_name>/*.png
        test/<class_name>/*.png

    NOTE: To get validation split with the FER-like signature,
          set train=False and balanced=True  -> uses 'valid'
          otherwise train=False uses 'test'.
    """
    def __init__(self, args, transform, train=True, idx=True, balanced=False,
                 imb_factor=1.0, debug=False, random_seed=42):
        super().__init__()
        self.root = _auto_base_dir(getattr(args, "dataset_path"), candidate_names=("CINIC-10", "cinic-10", "cinic10"))
        self.train = train
        self.transform = transform
        self.idx = idx

        if train:
            post = "train"
        else:
            post = "valid" if balanced else "test"

        split_dir = os.path.join(self.root, post)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"[CINIC10] split dir not found: {split_dir}")

        class_dirs = sorted([d for d in glob(os.path.join(split_dir, "*")) if os.path.isdir(d)])
        if len(class_dirs) == 0:
            raise RuntimeError(f"[CINIC10] no class folders under: {split_dir}")

        class_names = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.paths = []
        self.labels = []
        for c in class_names:
            cdir = os.path.join(split_dir, c)
            imgs = sorted([p for p in glob(os.path.join(cdir, "*")) if _is_image_file(p)])
            self.paths.extend(imgs)
            self.labels.extend([self.class_to_idx[c]] * len(imgs))

        self.labels = np.array(self.labels, dtype=np.int64)

        # optional debug limit
        self.paths, self.labels = _limit_debug(self.paths, self.labels, debug, n=5000, seed=random_seed)

        if imb_factor != 1.0 and train:
            self._make_imbalanced_dataset(imb_factor, random_seed)

        self.img_num_list = None
        self.get_img_num_per_cls()

        # Optional in-memory preload (same style as FER)
        self.pin_memory = getattr(args, "pin_memory", False)
        self.path_to_index = {p: i for i, p in enumerate(self.paths)}
        self.preloaded_images = None
        if self.pin_memory:
            self.preloaded_images = []
            for p in self.paths:
                img = Image.open(p).convert("RGB")
                img = img.copy()
                self.preloaded_images.append(img)

        self.get_macro_category()

    def get_macro_category(self):
        # CINIC-10 is basically balanced; use quantile-based fallback safely.
        counts = self.get_img_num_per_cls().astype(int)
        if len(counts) == 0:
            return {"many": [], "medium": [], "few": []}

        q67 = int(np.quantile(counts, 2 / 3))
        q33 = int(np.quantile(counts, 1 / 3))
        high, low = q67, q33

        categories = {"many": [], "medium": [], "few": []}
        for cls_idx, n in enumerate(counts):
            if n >= high:
                categories["many"].append(cls_idx)
            elif n <= low:
                categories["few"].append(cls_idx)
            else:
                categories["medium"].append(cls_idx)
        return categories

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx_):
        if self.preloaded_images is not None:
            img = self.preloaded_images[idx_]
        else:
            img = Image.open(self.paths[idx_]).convert("RGB")

        if isinstance(self.transform, list):
            samples = [tr(img) for tr in self.transform]
            return (samples, self.labels[idx_], idx_) if self.idx else (samples, self.labels[idx_])
        else:
            x = self.transform(img)
            return (x, self.labels[idx_], idx_) if self.idx else (x, self.labels[idx_])

    def _make_imbalanced_dataset(self, imb_factor, random_seed=42):
        rng = np.random.RandomState(random_seed)
        num_classes = len(np.unique(self.labels))
        img_max = max(np.sum(self.labels == c) for c in range(num_classes))
        img_num_list = []
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
            img_num_list.append(max(int(num), 1))

        new_paths, new_labels = [], []
        for cls_idx in range(num_classes):
            cls_indices = np.where(self.labels == cls_idx)[0]
            rng.shuffle(cls_indices)
            k = min(img_num_list[cls_idx], len(cls_indices))
            new_paths.extend([self.paths[i] for i in cls_indices[:k]])
            new_labels.extend([cls_idx] * k)

        self.paths = new_paths
        self.labels = np.array(new_labels, dtype=np.int64)

    def get_img_num_per_cls(self):
        if self.img_num_list is None:
            counter = Counter(self.labels)
            self.img_num_list = [0] * (np.max(self.labels) + 1)
            for key, value in counter.items():
                self.img_num_list[key] = value
        return np.array(self.img_num_list)


# -----------------------------
# 2) CORe50 (128x128 image folders)
# -----------------------------
class CORe50(Dataset):
    """
    CORe50 128x128 zip is typically organized by sessions and objects, e.g.
      <root>/
        s1/o1/*.png
        s1/o2/*.png
        ...
        s10/o50/*.png
        (sometimes s11 exists)

    Default split:
      train sessions = all except {s3, s7, s10}
      test  sessions = {s3, s7, s10}
    """
    _TEST_SESSIONS = {"s3", "s7", "s10"}

    def __init__(self, args, transform, train=True, idx=True, balanced=False,
                 imb_factor=1.0, debug=False, random_seed=42):
        super().__init__()
        self.root = _auto_base_dir(getattr(args, "dataset_path"), candidate_names=("core50_128x128", "core50"))
        self.train = train
        self.transform = transform
        self.idx = idx

        # Detect session dirs
        session_dirs = sorted([d for d in glob(os.path.join(self.root, "s*")) if os.path.isdir(d)])
        if len(session_dirs) == 0:
            # fallback: maybe already inside a session dir or flat structure
            session_dirs = [self.root]

        def _session_name(d):
            return os.path.basename(d)

        if len(session_dirs) > 1:
            if train:
                use_sessions = [d for d in session_dirs if _session_name(d) not in self._TEST_SESSIONS]
            else:
                use_sessions = [d for d in session_dirs if _session_name(d) in self._TEST_SESSIONS]
        else:
            use_sessions = session_dirs

        # Collect images; label from "/o{n}/" if possible
        self.paths = []
        self.labels = []

        o_pat = re.compile(r"(?:^|/)[oO](\d{1,3})(?:/|$)")
        for sdir in use_sessions:
            # expected: sdir/o*/img.png
            imgs = sorted(glob(os.path.join(sdir, "**", "*"), recursive=True))
            imgs = [p for p in imgs if os.path.isfile(p) and _is_image_file(p)]
            for p in imgs:
                m = o_pat.search(p.replace("\\", "/"))
                if m is None:
                    # fallback: try parse from filename like ..._o12_... (rare)
                    m2 = re.search(r"[oO](\d{1,3})", os.path.basename(p))
                    if m2 is None:
                        continue
                    obj_id = int(m2.group(1))
                else:
                    obj_id = int(m.group(1))

                # map to 0..49 if it's 1..50
                if 1 <= obj_id <= 50:
                    y = obj_id - 1
                else:
                    y = obj_id
                self.paths.append(p)
                self.labels.append(y)

        if len(self.paths) == 0:
            raise RuntimeError(
                f"[CORe50] No images found under {self.root}. "
                f"Expected s*/o*/ images. (Got sessions={len(session_dirs)})"
            )

        self.labels = np.array(self.labels, dtype=np.int64)

        # optional debug limit
        self.paths, self.labels = _limit_debug(self.paths, self.labels, debug, n=8000, seed=random_seed)

        if imb_factor != 1.0 and train:
            self._make_imbalanced_dataset(imb_factor, random_seed)

        self.img_num_list = None
        self.get_img_num_per_cls()

        # Optional in-memory preload (same style as FER)
        self.pin_memory = getattr(args, "pin_memory", False)
        self.path_to_index = {p: i for i, p in enumerate(self.paths)}
        self.preloaded_images = None
        if self.pin_memory:
            self.preloaded_images = []
            for p in tqdm(self.paths, disable=False):
                img = Image.open(p)
                img = img.copy()
                self.preloaded_images.append(img)

        self.get_macro_category()

    def get_macro_category(self):
        counts = self.get_img_num_per_cls().astype(int)
        if len(counts) == 0:
            return {"many": [], "medium": [], "few": []}

        q67 = int(np.quantile(counts, 2 / 3))
        q33 = int(np.quantile(counts, 1 / 3))
        high, low = q67, q33

        categories = {"many": [], "medium": [], "few": []}
        for cls_idx, n in enumerate(counts):
            if n >= high:
                categories["many"].append(cls_idx)
            elif n <= low:
                categories["few"].append(cls_idx)
            else:
                categories["medium"].append(cls_idx)
        return categories

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx_):
        if self.preloaded_images is not None:
            img = self.preloaded_images[idx_]
        else:
            img = Image.open(self.paths[idx_])

        if isinstance(self.transform, list):
            samples = [tr(img) for tr in self.transform]
            return (samples, self.labels[idx_], idx_) if self.idx else (samples, self.labels[idx_])
        else:
            x = self.transform(img)
            return (x, self.labels[idx_], idx_) if self.idx else (x, self.labels[idx_])

    def _make_imbalanced_dataset(self, imb_factor, random_seed=42):
        np.random.seed(random_seed)
        num_classes = len(np.unique(self.labels))
        img_num_list = []
        for cls_idx in range(num_classes):
            num = int(np.sum(self.labels == cls_idx) * (imb_factor ** (cls_idx / (num_classes - 1))))
            img_num_list.append(max(num, 1))

        new_paths, new_labels = [], []
        for cls_idx in range(num_classes):
            cls_indices = np.where(self.labels == cls_idx)[0]
            k = min(img_num_list[cls_idx], len(cls_indices))
            selected_indices = np.random.choice(cls_indices, size=k, replace=False)
            new_paths.extend([self.paths[i] for i in selected_indices])
            new_labels.extend([cls_idx] * k)

        self.paths = new_paths
        self.labels = np.array(new_labels, dtype=np.int64)

    def get_img_num_per_cls(self):
        if self.img_num_list is None:
            counter = Counter(self.labels)
            self.img_num_list = [0] * (np.max(self.labels) + 1)
            for key, value in counter.items():
                self.img_num_list[key] = value
        return np.array(self.img_num_list)


# -----------------------------
# 3) smallNORB (binary .mat files, not image files)
# -----------------------------
class SmallNORB(Dataset):
    """
    You downloaded the official files and gunzip'ed them, so you have:
      smallnorb-...-training-dat.mat
      smallnorb-...-training-cat.mat
      smallnorb-...-testing-dat.mat
      smallnorb-...-testing-cat.mat
      (+ info mat, optional)

    This dataset is stored in "binary matrix" format (NOT MATLAB .mat).
    We parse the header and read the tensor. See official description. :contentReference[oaicite:2]{index=2}

    Notes / knobs via args (optional):
      - args.norb_view in {"left","right","concat"}  (default="left")
        * left/right: one 96x96 view (grayscale)
        * concat: concatenates left|right horizontally -> 96x192 grayscale
      - args.norb_rgb (bool, default=False): convert grayscale PIL to RGB
    """

    _MAGIC_TO_DTYPE = {
        0x1E3D4C51: np.float32,  # single
        0x1E3D4C52: None,        # packed (not needed here)
        0x1E3D4C53: np.float64,  # double
        0x1E3D4C54: np.int32,    # int
        0x1E3D4C55: np.uint8,    # byte
        0x1E3D4C56: np.int16,    # short
    }

    def __init__(self, args, transform, train=True, idx=True, balanced=False,
                 imb_factor=1.0, debug=False, random_seed=42):
        super().__init__()
        self.root = _auto_base_dir(getattr(args, "dataset_path"), candidate_names=("smallnorb", "small_norb", "norb"))
        self.train = train
        self.transform = transform
        self.idx = idx

        self.norb_view = getattr(args, "norb_view", "left")  # left/right/concat
        self.norb_rgb = getattr(args, "norb_rgb", False)

        # locate files
        if train:
            dat_pat = os.path.join(self.root, "*training-dat.mat")
            cat_pat = os.path.join(self.root, "*training-cat.mat")
        else:
            dat_pat = os.path.join(self.root, "*testing-dat.mat")
            cat_pat = os.path.join(self.root, "*testing-cat.mat")

        dat_files = sorted(glob(dat_pat))
        cat_files = sorted(glob(cat_pat))
        if len(dat_files) == 0 or len(cat_files) == 0:
            raise FileNotFoundError(
                f"[SmallNORB] Missing dat/cat files in {self.root}\n"
                f"  searched: {dat_pat}\n  searched: {cat_pat}"
            )

        self.dat_path = dat_files[0]
        self.cat_path = cat_files[0]

        # load binary matrices into memory
        self.data = self._read_norb_binary_matrix(self.dat_path)   # expected shape: (24300, 2, 96, 96)
        self.labels = self._read_norb_binary_matrix(self.cat_path) # expected shape: (24300, 1) or (24300,)
        self.labels = np.array(self.labels).reshape(-1).astype(np.int64)

        if self.data.ndim != 4:
            raise RuntimeError(f"[SmallNORB] unexpected data ndim={self.data.ndim}, shape={self.data.shape}")

        # optional debug limit
        if debug:
            rng = np.random.RandomState(random_seed)
            n = min(5000, len(self.labels))
            sel = rng.choice(np.arange(len(self.labels)), size=n, replace=False)
            sel = np.sort(sel)
            self.data = self.data[sel]
            self.labels = self.labels[sel]

        # NOTE: imb_factor makes sense only on train split
        if imb_factor != 1.0 and train:
            self._make_imbalanced_dataset(imb_factor, random_seed)

        self.img_num_list = None
        self.get_img_num_per_cls()

        # keep FER-style fields (even though we don't have real file paths)
        self.paths = [f"{'train' if train else 'test'}_{i:05d}" for i in range(len(self.labels))]
        self.path_to_index = {p: i for i, p in enumerate(self.paths)}

        # pin_memory in FER meant "preload PIL"; here data already in RAM
        self.pin_memory = getattr(args, "pin_memory", False)
        self.preloaded_images = None

        self.get_macro_category()

    def _read_norb_binary_matrix(self, path: str):
        """
        Parse smallNORB binary matrix format:
          header:
            int32 magic
            int32 ndim
            int32 dim[3]  (and more dims if ndim>3)
          then data in little-endian with type encoded by magic.
        """
        with open(path, "rb") as f:
            # read magic, ndim, first 3 dims (little-endian int32)
            header = np.fromfile(f, dtype="<i4", count=5)  # magic, ndim, dim0, dim1, dim2
            if header.size < 5:
                raise RuntimeError(f"[SmallNORB] header too short: {path}")

            magic = int(header[0])
            ndim = int(header[1])
            d0, d1, d2 = int(header[2]), int(header[3]), int(header[4])

            dims = [d0, d1, d2]

            # read remaining dims if ndim > 3
            if ndim > 3:
                extra = np.fromfile(f, dtype="<i4", count=ndim - 3)
                dims.extend([int(x) for x in extra])

            dtype = self._MAGIC_TO_DTYPE.get(magic, None)
            if dtype is None:
                raise RuntimeError(f"[SmallNORB] unsupported/unknown magic={hex(magic)} in {path}")

            # read the remaining data
            count = int(np.prod(dims))
            data = np.fromfile(f, dtype=dtype, count=count)
            if data.size != count:
                raise RuntimeError(f"[SmallNORB] data truncated: expected {count}, got {data.size} from {path}")

            data = data.reshape(dims)
            return data

    def get_macro_category(self):
        counts = self.get_img_num_per_cls().astype(int)
        if len(counts) == 0:
            return {"many": [], "medium": [], "few": []}

        q67 = int(np.quantile(counts, 2 / 3))
        q33 = int(np.quantile(counts, 1 / 3))
        high, low = q67, q33

        categories = {"many": [], "medium": [], "few": []}
        for cls_idx, n in enumerate(counts):
            if n >= high:
                categories["many"].append(cls_idx)
            elif n <= low:
                categories["few"].append(cls_idx)
            else:
                categories["medium"].append(cls_idx)
        return categories

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx_):
        # data[idx_] shape: (2, 96, 96)
        pair = self.data[idx_]

        if self.norb_view == "right":
            arr = pair[1]
        elif self.norb_view == "concat":
            arr = np.concatenate([pair[0], pair[1]], axis=1)  # (96, 192)
        else:
            arr = pair[0]  # left default

        # ensure uint8 for PIL in 'L'
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
        if self.norb_rgb:
            img = img.convert("RGB")

        if isinstance(self.transform, list):
            samples = [tr(img) for tr in self.transform]
            return (samples, self.labels[idx_], idx_) if self.idx else (samples, self.labels[idx_])
        else:
            x = self.transform(img)
            return (x, self.labels[idx_], idx_) if self.idx else (x, self.labels[idx_])

    def _make_imbalanced_dataset(self, imb_factor, random_seed=42):
        rng = np.random.RandomState(random_seed)
        num_classes = len(np.unique(self.labels))
        img_max = max(np.sum(self.labels == c) for c in range(num_classes))
        img_num_list = []
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
            img_num_list.append(max(int(num), 1))

        keep_indices = []
        for cls_idx in range(num_classes):
            cls_indices = np.where(self.labels == cls_idx)[0]
            rng.shuffle(cls_indices)
            k = min(img_num_list[cls_idx], len(cls_indices))
            keep_indices.extend(cls_indices[:k].tolist())

        keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
        self.data = self.data[keep_indices]
        self.labels = self.labels[keep_indices]

    def get_img_num_per_cls(self):
        if self.img_num_list is None:
            counter = Counter(self.labels)
            self.img_num_list = [0] * (np.max(self.labels) + 1)
            for key, value in counter.items():
                self.img_num_list[key] = value
        return np.array(self.img_num_list)


class Large_dataset(Dataset):
    def __init__(self, root, train, transform=None, use_randaug=False):
        super().__init__()
        self.img_path = []
        self.labels = []
        self.transform = transform 
        self.use_randaug = use_randaug
        self.dataset_name='inat' if 'inat' in root else 'imagenet_lt'
        if 'imagenet_lt' in self.dataset_name:
            txt = 'ImageNet_LT_train.txt' if train else 'ImageNet_LT_test.txt' 
            with open(os.path.join(root, txt)) as f:
                for line in f:
                    img_path, label = line.split()
                    self.img_path.append(os.path.join(root, img_path.strip()))
                    self.labels.append(int(label.strip()))

        else:
            post = 'train' if train else 'val'
            json_path = os.path.join(root, f'{post}2018.json')
            with open(json_path, 'r') as f:
                dataset_json = json.load(f)

            images = dataset_json.get('images', [])
            annotations = dataset_json.get('annotations', [])

            image_id_to_path = {}
            for image_item in images:
                file_name = image_item.get('file_name')
                image_id = image_item.get('id')
                if file_name is None or image_id is None:
                    continue
                image_id_to_path[image_id] = os.path.join(root, file_name)

            for annotation in annotations:
                image_id = annotation.get('image_id')
                category_id = annotation.get('category_id')
                if image_id in image_id_to_path and category_id is not None:
                    self.img_path.append(image_id_to_path[image_id])
                    self.labels.append(int(category_id))

        self.targets = self.labels 
        self.img_num_list = self.get_img_num_per_cls()

    def get_img_num_per_cls(self):
        from collections import Counter
        counter = Counter(self.labels)
        num_classes = max(counter.keys())+1
        result = [0]*num_classes
        for key, value in counter.items():
            result[key] = value
        return np.array(result)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        path = self.img_path[idx]
        label = self.labels[idx]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if isinstance(self.transform, list):
                samples_list = [tr(sample) for tr in self.transform]
                return samples_list, label  # Unpack images as separate outputs
            else:
                sample = self.transform(sample)
        return sample, label 

    

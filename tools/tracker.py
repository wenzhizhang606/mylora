import os
from dotenv import load_dotenv, find_dotenv

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None


class _ExperimentTrackerBackend:
    """
    真正负责 wandb / swanlab 初始化、log、finish 的内部类。

    外部不直接使用这个类。
    外部只使用全局对象 tracker。
    """

    def __init__(
        self,
        project: str,
        name: str = None,
        config: dict = None,
        tracker_type: str = "swanlab",
        mode: bool = True,
    ):
        load_dotenv(find_dotenv())

        self.project = project
        self.name = name
        self.config = config or {}
        self.tracker_type = tracker_type.lower()
        self.mode = mode

        self.use_wandb = self.tracker_type == "wandb"
        self.use_swanlab = self.tracker_type == "swanlab"
        self.use_none = self.tracker_type == "none"

        if self.tracker_type not in ["wandb", "swanlab", "none"]:
            raise ValueError(
                f"不支持的 tracker_type: {tracker_type}, "
                f"可选值为: 'wandb', 'swanlab', 'none'"
            )

        self.initialized = False

    def _setup(self):
        if not self.mode or self.use_none:
            return

        if self.use_wandb:
            if wandb is None:
                raise ValueError("未安装 wandb 库!")

            wandb_key = os.getenv("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
            else:
                raise ValueError(".env 中未找到 WANDB_API_KEY")

        elif self.use_swanlab:
            if swanlab is None:
                raise ValueError("未安装 swanlab 库!")

            swanlab_key = os.getenv("SWANLAB_API_KEY")
            if swanlab_key:
                swanlab.login(api_key=swanlab_key)
            else:
                raise ValueError(".env 中未找到 SWANLAB_API_KEY")

    def init(self):
        if not self.mode or self.use_none:
            self.initialized = True
            return

        self._setup()

        if self.use_wandb:
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
            )
            print(f"Wandb initialized -> Project: {self.project}, Run: {self.name}")

        elif self.use_swanlab:
            swanlab.init(
                project=self.project,
                experiment_name=self.name,
                config=self.config,
            )
            print(
                f"SwanLab initialized -> Project: {self.project}, "
                f"Experiment: {self.name}"
            )

        self.initialized = True

    def log(self, metrics: dict, step: int = None):
        if not self.mode or self.use_none:
            return

        if not self.initialized:
            raise RuntimeError("Tracker 尚未初始化，请先调用 tracker.init(...)")

        if self.use_wandb:
            wandb.log(metrics, step=step)

        elif self.use_swanlab:
            swanlab.log(metrics, step=step)

    def finish(self):
        if not self.mode or self.use_none:
            return

        if not self.initialized:
            return

        if self.use_wandb:
            wandb.finish()

        elif self.use_swanlab:
            if hasattr(swanlab, "finish"):
                swanlab.finish()

        self.initialized = False


class _Tracker:
    """
    对外暴露的全局 tracker 门面对象。

    用法：

        from tracker import tracker

        tracker.init(...)
        tracker.log(...)
        tracker.finish()
    """

    def __init__(self):
        self._backend = None

    def init(
        self,
        project: str,
        name: str = None,
        config: dict = None,
        tracker_type: str = "swanlab",
        mode: bool = True,
    ):
        self._backend = _ExperimentTrackerBackend(
            project=project,
            name=name,
            config=config,
            tracker_type=tracker_type,
            mode=mode,
        )

        self._backend.init()

    def log(self, metrics: dict, step: int = None):
        if self._backend is None:
            raise RuntimeError("Tracker 尚未初始化，请先调用 tracker.init(...)")

        self._backend.log(metrics, step=step)

    def finish(self):
        if self._backend is None:
            return

        self._backend.finish()
        self._backend = None


# ============================================================
# 对外暴露的全局 tracker 对象
# ============================================================

tracker = _Tracker()


# ============================================================
# 兼容 import experiment_tracker as tracker 的写法
# ============================================================

def init(
    project: str,
    name: str = None,
    config: dict = None,
    tracker_type: str = "swanlab",
    mode: bool = True,
):
    return tracker.init(
        project=project,
        name=name,
        config=config,
        tracker_type=tracker_type,
        mode=mode,
    )


def log(metrics: dict, step: int = None):
    return tracker.log(metrics, step=step)


def finish():
    return tracker.finish()


def test():
    tracker.init(
        project="111",
        name="111",
        config={},
        tracker_type="wandb",
    )

    for i in range(100):
        tracker.log({"demo": i + 1}, step=i)

    tracker.finish()


if __name__ == "__main__":
    test()
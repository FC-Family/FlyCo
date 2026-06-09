class Tracker:
    def __init__(self, args) -> None:
        _ = args
        self.init_flag = False

    def reinit(self, frame, mask):
        raise NotImplementedError

    def if_init(self):
        return self.init_flag

    def init_tracker(self, frame, coords, labels=None, bbox=None):
        raise NotImplementedError

    def seg_acc_click(self, frame, coords, modes, multimask=True):
        raise NotImplementedError

    def track(
        self,
        frame,
        depth_mask=None,
        update_memory=True,
        check_lost=True,
        iou_threshold=0.7,
        update_mask=False,
    ):
        raise NotImplementedError

    def restart_tracker(self):
        raise NotImplementedError

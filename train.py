from ssd.modeling.detector.ssd_detector  import SSDDetector
from ssd.data.build import make_data_loader
from ssd.config.defaults import config as cfg




def train():
    # 数据加载
    max_iter = cfg.SOLVER.MAX_ITER // 1
    train_loader = make_data_loader(cfg, is_train=True, distributed=False, max_iter=max_iter, start_iter=1)
    
    for iteration, (images, targets, _) in enumerate(train_loader, 1):
        print(images)


def main():
    train()

    


if __name__ == "__main__":
    main()
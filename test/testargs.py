import argparse

p = argparse.ArgumentParser()
p.add_argument("--EPOCH", type=int, default=20)
p.add_argument("--BATCH_SIZE", type=int, default=6)
# p.add_argument("--CV", type=int, default=5)
# p.add_argument("--NUM_CLASS", type=int, default=4)
# p.add_argument("--NUM_CLASS", type=int, default=4)
# p.add_argument("--PARTIAL_TRAIN", action="store_true", default=False)
# p.add_argument("--PARTIAL_TRAIN_RATIO", type=float, default=0.003)
# p.add_argument("--NET_FREEZE", action="store_true", default=False)
# p.add_argument("--train_ratio", type=float, default=0.7)
# p.add_argument("--init_lr", type=float, default=1e-3)

args = p.parse_args()

print(args.EPOCH)
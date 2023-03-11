import argparse
from tasks import (
    task1_1,
    task1_2,
    task3,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PROJECT NAME. MCV-M6-Project. Team 2')

    parser.add_argument('--t1', action='store_true',
                        help='Task1 - produce noisy bounding boxes from GT annotations and calculate mIoU/AP')

    parser.add_argument('--t2', action='store_true',
                        help='Task2 - calculate mIoU/AP over time')

    parser.add_argument('--t3', action='store_true',
                        help='Task 3 - calculate MSEN, PEPN, and visualize errors in the estimated optical flow')

    parser.add_argument('--t4', action='store_true',
                        help='Task 4 - visualize optical flow')

    args = parser.parse_args()

    print(args)

    if args.t1:
        print('Launching Task 1')
        task1_2()

    if args.t2:
        print('Launching Task 2')
        t2_run()

    if args.t3:
        print('Launching Task 3')
        task3()

    if args.t4:
        print('Launching Task 4')
        t4_run()


import argparse
from tasks import (
    task1,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Road Traffic Monitoring Analysis for Video Surveillance. MCV-M6-Project. Team 2'
    )

    parser.add_argument('--t1', action='store_true', 
                        help='Task1 - background estimation with non-adaptive Gaussian model')

    parser.add_argument('--t2', action='store_true',
                        help='Task2 - background estimation with adaptive Gaussian model')

    parser.add_argument('--t3', action='store_true',
                        help='Task 3 - explore and evaluate a SOTA method')

    parser.add_argument('--t4', action='store_true',
                        help='Task 4 - background estimation with non-adaptive, color-aware, multidimensional Gaussian model')

    args = parser.parse_args()

    print(args)

    if args.t1:
        print('Launching Task 1')
        task1()

    # if args.t2:
    #     print('Launching Task 2')
    #     task1_2()
    #     task2()

    # if args.t3:
    #     print('Launching Task 3')
    #     task3()

    # if args.t4:
    #     print('Launching Task 4')
    #     t4_run()


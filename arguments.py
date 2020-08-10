import argparse

parser = argparse.ArgumentParser(description='XOR_FL')

parser.add_argument('--epochs', type=int, default=50000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2160, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.0001)

class Args():
    def __init__(self):
        self.args = parser.parse_args()

    def getParameters(self):
        return self.args

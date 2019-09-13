import torch
import sys

def diff(c1, c2):
    agg = 0
    for n1, n2 in zip(c1, c2):
        d1 = c1[n1].float()
        d2 = c2[n2].float()
        assert n1 == n2
        agg += torch.norm(d1 - d2, p=2)

    return agg.cpu().numpy()


def main():
    print('epoch, acc1, acc2, diff')
    for e in range(1, int(sys.argv[3])):
        c1 = torch.load('./'+sys.argv[1]+'/ckpt-i{}.pth'.format(e*10))
        c2 = torch.load('./'+sys.argv[2]+'/ckpt-i{}.pth'.format(e*10))

        assert c1['iter'] == c2['iter']
        print(c1['iter'], ",", c2['iter'], ",", diff(c1['state_dict'], c2['state_dict']))

if __name__ == "__main__":
    main()

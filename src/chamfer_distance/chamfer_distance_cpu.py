
import torch

from torch.utils.cpp_extension import load
cd = load(name="cd",
          sources=["chamfer_distance/chamfer_distance_cpu.cpp"
                   #,"chamfer_distance/chamfer_distance.cu"
                   ], verbose=True)

class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        # else:
        #     dist1 = dist1.cuda()
        #     dist2 = dist2.cuda()
        #     idx1 = idx1.cuda()
        #     idx2 = idx2.cuda()
        #     cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        # else:
        #     gradxyz1 = gradxyz1.cuda()
        #     gradxyz2 = gradxyz2.cuda()
        #     cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)



if __name__ == '__main__':


    import os 
    import sys
    sys.path.insert(1, os.path.abspath('.'))

    from utils.read_xyz import getpts_fromXYZ

    batch_size = 1
    filename = '/mnt/hdd1/mchiash2/PUdataset/all_testset/4/input/camel.xyz'
    filename2 = '/mnt/hdd1/mchiash2/PUdataset/all_testset/4/gt/camel.xyz'
    preds =  getpts_fromXYZ(filename)
    gts = getpts_fromXYZ(filename2)
    preds = torch.reshape(preds, (1,3, -1))
    gts = torch.reshape(gts, (1,3, -1))

    loss = ChamferDistance()

    dist1, dist2 = loss(preds,gts)

    train_loss = (torch.mean(dist1)) + (torch.mean(dist2))
    
    print(train_loss)
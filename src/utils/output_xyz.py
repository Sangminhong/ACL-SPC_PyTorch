import torch
import numpy as np
import open3d as o3d

def output_xyz(pts, output_path):
    """
    args
        pts: point cloud tensor to output [num_points, 3]
        output_path : 'directory/filename'



    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
    o3d.io.write_point_cloud(output_path, pcd)




if __name__ == '__main__':   
    a = torch.rand(100,3)
    output_xyz(a, '/home/mchiash2/PC_Upsampling/a.xyz')

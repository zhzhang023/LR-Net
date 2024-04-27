# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import numpy as np
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)
import matplotlib.pyplot as pyplot
# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, color=None,replace=True, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if color is not None:
        assert color.shape[0]==pc.shape[0]
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        if color is not None:
            return pc[choices], color[choices],choices
        else:
            return pc[choices], choices
    else:
        if color is not None:
            return pc[choices], color[choices]
        else:
            return pc[choices]

def random_sampling_pc_label(pc, label,num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], label[choices],choices
    else:
        return pc[choices], label[choices]
# ----------------------------------------
# Point Cloud/BBOX
# ----------------------------------------
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

# ----------------------------------------
# Point Cloud Jittering
# ----------------------------------------
# Jitter
def jitter_pointcloud(cloud_data, sigma=0.04, clip=0.05):
    #随机给点云增加噪声
    N, C = cloud_data.shape
    assert(clip > 0)
    # jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data=np.clip(np.random.normal(0.0, scale=sigma, size=(N, C)),
            a_min=-1*clip, a_max=clip)
    jittered_data += cloud_data
    return jittered_data
# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b, :, :], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)


def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize, vsize, vsize, num_sample, 3))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i, j, k) not in loc2pc:
                    vol[i, j, k, :, :] = np.zeros((num_sample, 3))
                else:
                    pc = loc2pc[(i, j, k)]  # a list of (3,) arrays
                    pc = np.vstack(pc)  # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0] > num_sample:
                        pc = random_sampling(pc, num_sample, False)
                    elif pc.shape[0] < num_sample:
                        pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i, j, k]) + 0.5) * voxel - radius
                    pc = (pc - pc_center) / voxel  # shift and scale
                    vol[i, j, k, :, :] = pc
    return vol


def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b, :, :], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2 * radius / float(imgsize)
    locations = (points[:, 0:2] + radius) / pixel  # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n, :])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n, :])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i, j) not in loc2pc:
                img[i, j, :, :] = np.zeros((num_sample, 3))
            else:
                pc = loc2pc[(i, j)]
                pc = np.vstack(pc)
                if pc.shape[0] > num_sample:
                    pc = random_sampling(pc, num_sample, False)
                elif pc.shape[0] < num_sample:
                    pc = np.lib.pad(pc, ((0, num_sample - pc.shape[0]), (0, 0)), 'edge')
                pc_center = (np.array([i, j]) + 0.5) * pixel - radius
                pc[:, 0:2] = (pc[:, 0:2] - pc_center) / pixel
                img[i, j, :, :] = pc
    return img


# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def write_ply_color(points, labels, filename, num_classes=2, colormap=pyplot.cm.jet):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels) + 1
    else:
        assert (num_classes > np.max(labels))

    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    colors = [colormap(i / float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        if labels[i]==0:
            c = [131,175,155]
        else:
            c = [254, 67, 101]
        # c = colors[labels[i]]
        # c = [int(x * 255) for x in c]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def write_ply_rgb(points, colors,out_filename,lab=None,one_color=False):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    # colors = colors.astype(int)
    # N = points.shape[0]
    # fout = open(out_filename, 'w')
    # for i in range(N):
    #     c = colors[i, :]
    #     fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    # fout.close()
    # colors = colors.astype(int)
    N = points.shape[0]
    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        if one_color:
            c = colors
        else:
            c=colors[i]
        if lab is not None and lab[i]==1:
            c = [254, 67, 101]
        vertex.append((points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(out_filename)

def write_ply_rgb_twopc(points_1, colors_1, points_2, colors_2,out_filename):
    if colors_1==None or colors_2==None:
        colors_1=np.zeros_like(points_1).astype(int)
        colors_2 = np.zeros_like(points_2).astype(int)
        for i in range(points_1.shape[0]):
            colors_1[i,0]=255
            colors_1[i, 1] = 0
            colors_1[i, 2] = 0

        for j in range(points_2.shape[0]):
            colors_2[j,0]=0
            colors_2[j, 1] = 255
            colors_2[j, 2] = 0
    else:
        colors_1 = colors_1.astype(int)
        colors_2 = colors_2.astype(int)
    N_1 = points_1.shape[0]
    N_2 = points_2.shape[0]
    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N_1):
        c = colors_1[i]
        vertex.append((points_1[i, 0], points_1[i, 1], points_1[i, 2], c[0], c[1], c[2]))
    for i in range(N_2):
        c = colors_2[i]
        vertex.append((points_2[i, 0], points_2[i, 1], points_2[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(out_filename)

def write_ply_rgb_twopc_2(points_1, colors_1, points_2, colors_2,out_filename):

    colors_1 = colors_1.astype(int)
    colors_2 = colors_2.astype(int)
    N_1 = points_1.shape[0]
    N_2 = points_2.shape[0]
    vertex = []
    # colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N_1):
        c = colors_1[i]
        vertex.append((points_1[i, 0], points_1[i, 1], points_1[i, 2], c[0], c[1], c[2]))
    for i in range(N_2):
        c = colors_2[i]
        vertex.append((points_2[i, 0], points_2[i, 1], points_2[i, 2], c[0], c[1], c[2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(out_filename)

# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111,projection='3d')
    z_max=np.max(points[:,2])
    z_min=np.min(points[:,2])
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    Blue=np.ones_like(points[:,2])-((points[:,2]-z_min)/(z_max-z_min))
    Red = ((points[:, 2] - z_min) / (z_max - z_min))
    Green=np.zeros_like(Blue)
    color=np.stack((Red,Green,Blue),axis=1)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               cmap='spectral',
           c=color,
           s=18,
           linewidth=0,
           alpha=1,
           marker=".")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    ax.set_aspect(aspect='auto',adjustable='datalim')
    # plt.show()
    plt.savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------
def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix


def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


# ----------------------------------------
# BBox
# ----------------------------------------
def bbox_corner_dist_measure(crnr1, crnr2):
    """ compute distance between box corners to replace iou
    Args:
        crnr1, crnr2: Nx3 points of box corners in camera axis (y points down)
        output is a scalar between 0 and 1
    """

    dist = sys.maxsize
    for y in range(4):
        rows = ([(x + y) % 4 for x in range(4)] + [4 + (x + y) % 4 for x in range(4)])
        d_ = np.linalg.norm(crnr2[rows, :] - crnr1, axis=1).sum() / 8.0
        if d_ < dist:
            dist = d_

    u = sum([np.linalg.norm(x[0, :] - x[6, :]) for x in [crnr1, crnr2]]) / 2.0

    measure = max(1.0 - dist / u, 0)
    print(measure)

    return measure


def point_cloud_to_bbox(points):
    """ Extract the axis aligned box from a pcl or batch of pcls
    Args:
        points: Nx3 points or BxNx3
        output is 6 dim: xyz pos of center and 3 lengths
    """
    which_dim = len(points.shape) - 2  # first dim if a single cloud and second if batch
    mn, mx = points.min(which_dim), points.max(which_dim)
    lengths = mx - mn
    cntr = 0.5 * (mn + mx)
    return np.concatenate([cntr, lengths], axis=which_dim)




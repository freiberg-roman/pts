import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as FV


def get_heightmap(
    points,
    colors,
    cam_pose,
    workspace_limits,
    heightmap_resolution,
):
    # Compute heightmap size
    heightmap_size = np.round(
        (
            (workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
            (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution,
        )
    ).astype(int)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    points = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(points))
        + np.tile(cam_pose[0:3, 3:], (1, points.shape[0]))
    )

    # Sort surface points by z value
    sort_z_ind = np.argsort(points[:, 2])
    points = points[sort_z_ind]
    colors = colors[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(
        np.logical_and(
            np.logical_and(
                np.logical_and(
                    points[:, 0] >= workspace_limits[0][0],
                    points[:, 0] < workspace_limits[0][1],
                ),
                points[:, 1] >= workspace_limits[1][0],
            ),
            points[:, 1] < workspace_limits[1][1],
        ),
        points[:, 2] < workspace_limits[2][1],
    )

    points = points[heightmap_valid_ind]
    colors = colors[heightmap_valid_ind]
    colors = np.uint8(colors * 255)

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros(
        (heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8
    )
    color_heightmap_g = np.zeros(
        (heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8
    )
    color_heightmap_b = np.zeros(
        (heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8
    )
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor(
        (points[:, 0] - workspace_limits[0][0]) / heightmap_resolution
    ).astype(int)
    heightmap_pix_y = np.floor(
        (points[:, 1] - workspace_limits[1][0]) / heightmap_resolution
    ).astype(int)
    color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = colors[:, [0]]
    color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = colors[:, [1]]
    color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = colors[:, [2]]
    color_heightmap = np.concatenate(
        (color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2
    )
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = points[:, 2]

    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = 0
    return color_heightmap, depth_heightmap


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FV.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

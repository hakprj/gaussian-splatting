import open3d as o3d

# Load the PLY
pcd = o3d.io.read_point_cloud("/mnt/c/Users/hayra/Downloads/GP_360_Large_GP_360_Larg_30K_imags_masked_zero_scale_with_2_bg_penalty/content/GP_360_Large_GP_360_Larg_30K_imags_masked_zero_scale_with_2_bg_penalty/point_cloud/iteration_30000/point_cloud.ply")

# Estimate normals (required for Poisson)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(100)

# Poisson reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Remove low-density vertices (cleans up outer noise)
import numpy as np
vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)

o3d.io.write_triangle_mesh("mesh_output.ply", mesh)
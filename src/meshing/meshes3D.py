import pyvista as pv
from skimage import measure
import numpy as np
from stl import mesh #pip install numpy-stl
import igl
import wildmeshing as wm
import os
import pymeshlab
from typing import Optional, Type, Union, Tuple

class surfaceMesh:
    """Creates a triangle surface mesh in 3D.
    Segments assigned 1 in mask are considered inside the object.
        :param mask: mask of the moving image (dim1,dim2,dim3)
        :type mask: torch.Tensor
        :param tempDir: path for temporary mesh storage
        :type tempDir: str
        :param faceNumbers: number of faces required; if None no reduction in faces
        :type faceNumbers: int
        :param level: level of iso surface for marching cubes
        :type level: float"""
    def __init__(self, mask: np.array, tempDir: str, faceNumbers: Optional[int]=None, level: Optional[float]=0.8):
        "Constructor method."
        self.faceNumbers = faceNumbers
        self.tempDir = tempDir
        self.mask = mask
        self.level = level

    def triangulate(self)->Tuple[np.array, np.array]:
        """Creates a triangle surface mesh.
        :param level: level of iso surface for marching cubes
        :type level: float"""
        mask = np.where(self.mask == 1, 1, 0)
        vertices, faces, normals, values = measure.marching_cubes(mask, level=self.level)
        faces_new = np.array(faces.copy())
        faces_new[:, 1] = faces[:, 2]
        faces_new[:, 2] = faces[:, 1]
        faces = faces_new
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j], :]
        cube.save(os.path.join(self.tempDir, f"surface_mesh_temp.stl"))
        if self.faceNumbers is not None:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(
                os.path.join(self.tempDir, f"surface_mesh_temp.stl"))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=self.faceNumbers)
            ms.save_current_mesh(
                os.path.join(self.tempDir, f"surface_mesh_temp.stl"))
        else:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(
                os.path.join(self.tempDir, f"surface_mesh_temp.stl"))
            ms.meshing_decimation_quadric_edge_collapse()
            ms.save_current_mesh(
                os.path.join(self.tempDir, f"surface_mesh_temp.stl"))
        vertices, faces = igl.read_triangle_mesh(
            os.path.join(self.tempDir, f"surface_mesh_temp.stl"), 'float')
        return vertices, faces

    def getVerticesAndSimplicesSurface(self):
        """Returns the vertices and simplices of the surface mesh. If reuse, also saves the mesh.
            :return vertices: vertices (# vertices, 3)
            :rtype vertices: np.array
            :return simplices: simplices (#simplices,3)
            :rtype simplices: np.array"""
        vertices, faces = self.triangulate()
        return vertices,faces


class tetrahedralMesh:
    """Creates a tetrahedral mesh in 3D.
    Can create the mesh either from a segmented mask image, or an existing 3D stl surface mesh.
    If creating from a segmented mask image, will create the surface mesh on the fly.
    Segments assigned segmentLabel in mask are considered inside the object.
    """
    def __init__(self):
        "Constructor method."

    def getTetrahedralMeshFromMask(self,mask: np.array, tempDir: str, faceNumbers: Optional[int]=None, level: Optional[float]=0.8)->Tuple[np.array,np.array]:
        """Creates a tetrahedral mesh from a mask image. Creates a surface mesh first.
        :param mask: mask of the moving image (dim1,dim2,dim3)
        :type mask: np.array
        :param tempDir: path for temporary mesh storage
        :type tempDir: str
        :param faceNumbers: number of faces required; if None no reduction in faces
        :type faceNumbers: int
        :param level: level of iso surface for marching cubes
        :type level: int
        :return volVertices: vertices (#simplices,3)
        :rtype volVertices: np.array
        :return volTets: simplices (#simplices,4)
        :rtype volTets: np.array"""
        c=surfaceMesh(mask, tempDir,  faceNumbers,  level)
        vertices,faces=c.getVerticesAndSimplicesSurface()
        volVertices, volSimplices = self.tetrahedralize(vertices, faces)
        return volVertices,volSimplices

    def getTetrahedralMeshFromSurface(self,pathToMesh: str,segmentName: int)->Tuple[np.array,np.array]:
        """Creates a tetrahedral mesh from a triangle surface.
        :param pathToMesh: path to stl mesh
        :type pathToMesh: str
        :param segmentName: label of segmentation to be meshed
        :type segmentName: int
        :return volVertices: vertices (#simplices,3)
        :rtype volVertices: np.array
        :return volTets: simplices (#simplices,4)
        :rtype volTets: np.array"""
        vertices, faces = igl.read_triangle_mesh(
            os.path.join(pathToMesh), 'float')
        volVertices, volSimplices = self.tetrahedralize(vertices, faces)
        return volVertices,volSimplices

    def tetrahedralize(self,vertices: np.array,simplices: np.array)->Tuple[np.array,np.array]:
        """Creates a tetrahedral mesh.
            :param vertices: vertices of triangle surface mesh (#simplices,3)
            :type vertices: np.array
            :param simplices: simplices of triangle surface mesh (#simplices,3)
            :type simplices: np.array
            :return volVertices: vertices (#simplices,3)
            :rtype volVertices: torch.Tensor
            :return volTets: simplices (#simplices,4)
            :rtype volTets: torch.tensor"""
        tetra = wm.Tetrahedralizer()
        tetra.set_mesh(vertices, simplices)
        tetra.tetrahedralize()
        volVertices, volTets = tetra.get_tet_mesh()
        volVertices, volTets, _, _ = igl.remove_unreferenced(volVertices, volTets)
        return volVertices, volTets

    def getExteriorPoints(self,vertices: np.array,simplices: np.array)->np.array:
         """Returns the exterior vertices of the tetrahedra mesh.
            :param vertices: vertices (# vertices, 3)
            :type vertices: np.array
            :param simplices: simplices (#simplices,4)
            :type simplices: np.array
            :return vertices: vertices (# vertices, 3)
            :rtype vertices: np.array"""
         vertices=vertices
         simplices=simplices
         cells = np.hstack([np.full((simplices.shape[0], 1), 4), simplices]).flatten()
         cell_types = np.full(simplices.shape[0], pv.CellType.TETRA, dtype=np.uint8)
         grid = pv.UnstructuredGrid(cells, cell_types, vertices)
         surface = grid.extract_surface()
         exterior_points = surface.points
         return exterior_points



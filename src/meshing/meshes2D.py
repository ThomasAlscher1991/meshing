from skimage import measure
import numpy as np
import scipy
from typing import Optional, Type, Union, Tuple

class triangleMesh:
    """Class for creating triangle meshes in 2D by Delaunay triangulation.
    """
    def __init__(self):
        """Constructor method."""
        self.vertices = None
        self.simplices=None

    def getTriangleMeshFromMask(self,mask: np.array,contourLevel: Optional[float]=0.5):
        """Generates the triangle mesh from a masked image.
        Segments assigned 1 in mask are considered inside the object.
        :param mask: mask of the moving image (dim1,dim2)
        :type mask: np.array
        :param segmentLabel: label of segmentation to be meshed
        :type segmentLabel: int
        :param contourLevel: level for contour
        :type contourLevel: float"""
        mask = np.where(mask == 1, 1,0)
        contours = measure.find_contours(mask, level=contourLevel)
        contoursFused = np.empty((0, contours[0].shape[1]))
        for c in contours:
            contoursFused = np.concatenate((contoursFused, c))
        self.vertices=contoursFused
        self.simplices = scipy.spatial.Delaunay(self.vertices).simplices

    def getTriangleMeshfromContour(self,contour: np.array):
        """Generates the triangle mesh from a contour.
        :param contour: list of 2d coordinates
        :type contour: torch.Tensor"""
        self.vertices = contour
        self.simplices = scipy.spatial.Delaunay(self.vertices).simplices

    def divide(self,maxEdgeLength: float):
        """Iteratively divides edges to conform to maxEdgeLength.
            :param maxEdgeLength: maximum length of edges
            :type maxEdgeLength: float"""
        vertices=self.vertices
        switch=True
        while switch:
            simplices = scipy.spatial.Delaunay(vertices).simplices
            c1=vertices[simplices[:,0]]-vertices[simplices[:,1]]
            edgeLength1=np.linalg.norm(c1,axis=1)
            c1=vertices[simplices[:,1]][edgeLength1>maxEdgeLength]+c1[edgeLength1>maxEdgeLength]*0.5
            c2=vertices[simplices[:,1]]-vertices[simplices[:,2]]
            edgeLength2=np.linalg.norm(c2,axis=1)
            c2=vertices[simplices[:,2]][edgeLength2>maxEdgeLength]+c2[edgeLength2>maxEdgeLength]*0.5
            c3=vertices[simplices[:,2]]-vertices[simplices[:,0]]
            edgeLength3=np.linalg.norm(c3,axis=1)
            c3=vertices[simplices[:,0]][edgeLength3>maxEdgeLength]+c3[edgeLength3>maxEdgeLength]*0.5
            if len(c1)==0 and len(c2)==0 and len(c3)==0:
                switch=False
            else:
                vertices=np.concatenate((vertices,c1,c2,c3))
        self.vertices=vertices
        self.simplices = scipy.spatial.Delaunay(self.vertices).simplices

    def getVerticesAndSimplices(self)-> Tuple[np.array,np.array]:
        """Returns the vertices and simplices of the mesh as torch tensors.
            :return vertices: vertices (# vertices, 2)
            :rtype vertices: np.array
            :return simplices: simplices (#simplices,3)
            :rtype simplices: np.array"""
        return self.vertices,self.simplices

    def checkCrossing(self,p0: np.array,p1: np.array,samplerate: int,mask: np.array,kickValue: Optional[float]=0):
        """Checks if the edge is crossing non segmented areas of the mask.
            :param p0: end point 1 of edge
            :type p0: np.array
            :param p1: end point 2 of edge
            :type p1: np.array
            :param mask: image mask
            :type mask: np.array
            :param kickValue: value below or equal to which a point on the line is considered outside
            :type kickValue: float
            :param samplerate: how many evaluation points are created on the edge
            :type samplerate: int
            :return switch: reduced simplices
            :rtype switch: torch.Tensor
        """
        x = np.arange(0, mask.shape[0], 1)
        y = np.arange(0, mask.shape[1], 1)
        f = scipy.interpolate.interp2d(x, y, mask.T, kind='linear')
        switch=False
        if samplerate==1 or samplerate==2:
            val0=f(p0[0],p0[1])
            val1=f(p1[0],p1[1])
            if val0<=kickValue or val1<=kickValue:
                switch=True
        else:
            xh=np.max([p0[0],p1[0]])
            xl=np.min([p0[0],p1[0]])
            yh=np.max([p0[1],p1[1]])
            yl=np.min([p0[1],p1[1]])
            samplex = np.linspace(xl, xh, samplerate)[1:-1]
            sampley = np.linspace(yl, yh, samplerate)[1:-1]
            for xnew,ynew in zip(samplex,sampley):
                val=f(xnew,ynew)
                if val<=kickValue:
                    switch=True
        return switch

    def cleanConvex(self,tri: np.array,points: np.array,mask: np.array):
        """
        Reduces the convex mesh to necessary vertices and simplices by deleting simplices that lie fully or partially outside of the segmented region.
        Subsequently deletes unused vertices.
            :param tri: simplices
            :type tri: np.array
            :param points: vertices
            :type points: np.array
            :param mask: image mask
            :type mask: np.array
        """
        decision=np.ones((tri.shape[0]),dtype='int')
        for cnt,t in enumerate(tri):
            edge1=self.checkCrossing(points[t[0]],points[t[1]],3,mask)
            edge2=self.checkCrossing(points[t[1]],points[t[2]],3,mask)
            edge3=self.checkCrossing(points[t[2]],points[t[0]],3,mask)
            if edge1 or edge2 or edge3:
                decision[cnt]=0
        tri=tri[decision==1]
        vertexIDsUsed = np.unique(tri)
        vertexIDs = np.arange(0, points.shape[0])
        vertexIDsNotUsed = vertexIDs[np.isin(vertexIDs, vertexIDsUsed) == False]
        for cnt in range(vertexIDsNotUsed.shape[0]):
            vID = vertexIDsNotUsed[cnt]
            before = points[:vID, :]
            after = points[vID + 1:, :]
            points = np.concatenate((before, after))
            tri = np.where(tri > vID, tri - 1, tri)
            vertexIDsNotUsed = np.where(vertexIDsNotUsed > vID, vertexIDsNotUsed - 1, vertexIDsNotUsed)
        self.simplices = tri
        self.vertices = points

    def getTriangleMeshFromMaskSeeding(self,mask: np.array,contourLevel: Optional[float]=0.5,number: Optional[int]=100000):
        """Generates the triangle mesh from a masked image and seeds random vertices inside the mesh.
        :param mask: mask of the moving image (dim1,dim2)
        :type mask: torch.tensor
        :param contourLevel: level for contour
        :type contourLevel: float
        :param number: number of all points, seeded and contour
        :type number: int"""
        mask = np.where(mask == 1, 1,0)
        contours = measure.find_contours(mask, level=contourLevel)
        contoursFused = np.empty((0, contours[0].shape[1]))
        for c in contours:
            contoursFused = np.concatenate((contoursFused, c))
        if contoursFused.shape[0]>number:
            pass
        else:
            fill=number-contoursFused.shape[0]
            indices = np.indices(mask.shape)
            pts = np.empty((np.prod(mask.shape), len(mask.shape)))
            for i, slide in enumerate(indices):
                pts[:, i] = slide.flatten()
            pts=pts[mask.flatten()==1]
            random_tensor = np.randperm(pts.shape[0])[:fill]
            pts = pts[random_tensor]
            contoursFused=np.concatenate((contoursFused,pts))
        self.vertices=contoursFused
        self.simplices = scipy.spatial.Delaunay(self.vertices).simplices






"""
Matrix To STL

Author: mosselini1 (https://www.printables.com/@mosselini1_1346202)
Date: 07/01/23
"""

import os,copy
import numpy as np
from stl import mesh

def default_cube_info():
	# Define the 8 vertices of the cube (x [=width],y [=depth],z [=height])
	vertices = np.array([\
			[0, 0, 0], #0
			[1, 0, 0], #1
			[1, 1, 0], #2
			[0, 1, 0], #3
			[0, 0, 1], #4
			[1, 0, 1], #5
			[1, 1, 1], #6
			[0, 1, 1]])#7

	# Define the 6 faces of the the cube
	faces = np.array([ # from bottom-left anti-clockwise [=bl,br,tr,tl]
		[0,1,5,4], #
		[1,2,6,5], #
		[4,5,6,7], #
		[0,1,2,3], #
		[3,2,6,7], #
		[0,3,7,4], #
	])

	return vertices, faces

def arr_pos_in_arr_list(list_np_arrays,array_to_check):
	if list_np_arrays is not None and list_np_arrays.size != 0:
		temp = np.where((list_np_arrays == array_to_check).all(axis=1))[0]
		res = [None] if temp.size == 0 else temp
	else:
		res = [None]
	
	return res

def add_cube(vertices, faces, scale=1, begin_pos=None):
	
	if begin_pos is None: begin_pos = np.array([0,0,0])
	
	new_v, dflt_f = default_cube_info()
	new_f = copy.deepcopy(dflt_f)
	
	new_v *= scale
	new_v += begin_pos

	beg_pos = faces.max()+1 if faces is not None else 0
	
	to_add = []
	for v in range(new_v.shape[0]):
		coord = new_v[v,:]
		
		pos = arr_pos_in_arr_list(vertices,coord)[0]
		if pos is None:
			pos = beg_pos
			beg_pos += 1
			vertices = np.vstack((vertices,coord)) if vertices is not None else np.array((copy.deepcopy(coord),))
				
		new_f[dflt_f==v] = pos
	
	# concatenate the faces arrays
	faces = np.concatenate((faces,new_f)) if faces is not None else new_f
	# duplicated faces for the separation of different objects
	
	return vertices, faces

def matrix_to_coords(mat):
	res = []
	lay,row,col = mat.shape
	for l in range(lay):
		for r in range(row):
			for c in range(col):
				if mat[l,r,c] != 0:
					res.append([c,row-1-r,lay-1-l])
	return res

def coords_to_elements(coords,scale=1):
	vertices, faces = None,None
	for c,coord in enumerate(coords):
		vertices, faces = add_cube(vertices, faces, scale , np.array(coord)*scale)
	return vertices, faces

def faces_triangulation(faces):
	# cut faces in 2
	triangle_faces = []
	for f in range(faces.shape[0]):
		face = faces[f]
		triangle_faces.append(np.array([face[0],face[1],face[2]]))
		triangle_faces.append(np.array([face[0],face[2],face[3]]))
	
	return np.array(triangle_faces)

def mesh_creation(vertices, faces):
	# Create the mesh
	model = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(faces):
		for j in range(3):
			model.vectors[i][j] = vertices[f[j],:]
	return model

def main(mat,scale,outpath):
	coords = matrix_to_coords(mat)
	vertices, faces = coords_to_elements(coords,scale)
	triangle_faces = faces_triangulation(faces)
	model = mesh_creation(vertices, triangle_faces)

	# Write the mesh to file
	model.save(outpath)

if __name__ == "__main__": 
	
	mat = np.array([
	[[0,0,0],
	[0,1,0],
	[0,1,1]],
	[[0,0,0],
	[0,1,0],
	[1,1,1]]
	])
	
	scale = 10 #equals the edge size of the cubes
	
	outpath = os.path.join(os.getcwd(),"model.stl")
	
	main(mat,scale,outpath)

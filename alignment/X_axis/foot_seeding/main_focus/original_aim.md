We have a person in adam's bent position, we took the video of the person in 360 from a far view so that the person's foot, hands back and head are visible and the ground. 

Video->frames->VGGT->denoising->surface_mesh using meshlab. 


Now there is issue of axis alignment. The X,Y,Z axis are not aligned with the person's body. So we need to align the X axis with the person's body. 

Due to VGGT usage, the axes are as per first frame reference. 

I have sucessfully computed the ground plane from the mesh and aligned the Y axis to be the negative normal of the ground plane i.e. feets are at max Y and head/torso region at min Y. 

Aim: To align the X axis parallel to the line joining the points where the 2 feet meet the ground plane. 

Idea: 
We plan to build a gradio based GUI that can do this task. 

1. Input: mesh in ply format and json file that has the ground plane information. 

2. Process of estimation: 

2.1 The input mesh which is a surface mesh needs to be converted into a volumetric mesh. For this we can use voxelisation and have voxelized mesh. save this metafile voxelized_mesh or any other method that you have 

2.2 Now from this voxelized mesh, using relative distances in terms of a parameter h. 

h is the maximum bounding distance in the y-axis or more precisely from the ground plane. It is the maximum distance of the mesh point that can be there from the ground plane. 

Using this h we will represent all of our intrisic parameters of the algorithim so that it becomes scale-invariant. 

2.3. We will segment out a roi from voxelized mesh at a height of 0.05h to 0.25 h. keep options of tuning this parameters in the gradio UI. save the roi.ply file as metadata. So this roi will contain the legs mostly and hands could be there. 

2.4 Now after this our aim to 3D skeltonize this volume of ROI . 

2.5 Now there would be 4 roughly skeleton-lines corresponding to two hands and two legs. Our aim is to take only leg skeleton lines and fit a line to the skeleotn-line of the leg and extend it to the ground plane and from that the poinst where these center line intersect the ground plane we will join those precise points and orient our x-axis in that direction. 

For removing hand we can use logic that leg-skeleton line would be nearer to ground plane rather than hand-skeleton lines. Make sure to save the roi_skeleton.ply 
roi_skeleton_prunned.ply 
and a file named aligned_mesh.ply 
and aligned_mesh_ground_points.ply[text](pca_results)
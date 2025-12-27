function [depth]=initialization_for_NRSfM_local_all()
load('E:/Python_code/ECCV/pythonProject3/NRSfM_dataset/Kinect/matlab.mat');
addpath('./BBS/');%Use tool in BBS toolpack (Bicubic B-Splines)
num=size(scene.m,2);
for i=1:1:num
    normalized_Image{i}.point_2d=scene.m(i).m(1:2,:);%2D value on image as input
end
point_size=size(normalized_Image{1}.point_2d,2);
for i=1:1:num
     point{i}=[];
     point_used{i}=[1:1:point_size;];
end
for i=1:1:num-1
     measurements{1,i}.image=[1,i+1];
     measurements{1,i}.point=[1:point_size;1:point_size]';
end
[X_update_k]=LLS11(J,normalized_Image,measurements,[],[],point);
x0new1=-X_update_k;
x_1=x0new1(1:2:end);
x_2=x0new1(2:2:end);
[depth]=depth_recovery_python(x_1,x_2,num,normalized_Image,point_used);
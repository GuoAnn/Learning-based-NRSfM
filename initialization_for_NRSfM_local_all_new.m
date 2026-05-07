function [depth]=initialization_for_NRSfM_local_all(file_name)
load(file_name);
addpath('./BBS/');
num=size(scene.m,2);

%% 1. 准备原始数据
for i=1:1:num
    normalized_Image{i}.point_2d=scene.m(i).m(1:2,:);
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

%% 2. 调用LLS11进行联合优化
% lambda_reg: 第一帧平面正则化权重, 越大帧1越趋向平面
lambda_reg = 1e2;
[X_update_k]=LLS11(J,normalized_Image,measurements,[],[],point,lambda_reg);

x0new1=-X_update_k;
x_1=x0new1(1:2:end);
x_2=x0new1(2:2:end);

%% 3. 深度恢复
[depth]=depth_recovery_python(x_1,x_2,num,normalized_Image,point_used);

end
function [P_grid_neq]=depth_recovery_python(x_1,x_2,num,normalized_Image,point_used)
k1_all=x_1;
k2_all=x_2;
% k1_all=x0new(1:2:end);
% k2_all=x0new(2:2:end);
k1_all=reshape(k1_all,size(k1_all,1)/num,num)';
k2_all=reshape(k2_all,size(k2_all,1)/num,num)';
% depth=reshape(depth,size(depth,1)/num,num)';
% rmd=reshape(rmd,size(rmd,1)/(num-1),num-1)';
% for i=1:1:num
%     Estimated{i}.point_3d=[normalized_Image{i}.point_2d;ones(1,size(normalized_Image{i}.point_2d,2))].*depth(i,:);
%     Pgth(3*(i-1)+1:3*(i-1)+3,:) =  Ground_truth{i}.point_3d;%scene.Pgth(i).P;
%     [~,qw,~]= absor(Estimated{i}.point_3d,Pgth(3*(i-1)+1:3*(i-1)+3,:),'doScale',true);
%     P2(3*(i-1)+1:3*(i-1)+3,:) = qw;
% end
u_all=[];
v_all=[];
for i=1:1:num
    u_all = [u_all; normalized_Image{i}.point_2d(1,:)];
    v_all = [v_all; normalized_Image{i}.point_2d(2,:)];
end
N1 =-k1_all;
N2 =-k2_all;
N3 = 1-u_all.*N1-v_all.*N2;
n = sqrt(N1.^2+N2.^2+N3.^2);
N1 = N1./n ; N2 = N2./n; N3 = N3./n;
N = [N1(:),N2(:),N3(:)]';
N_res = reshape(N(:),3*num,size(u_all,2));
% % Integrate normals to find depth
P_grid=calculate_depth(N_res,u_all,v_all,1e0,point_used);
P_grid_neq=depth_adjust_by_scale(P_grid,10);
% compare with ground truth
end

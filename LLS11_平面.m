function [X_update_k]=LLS11(J,normalized_Image,measurements,k1_all,k2_all,point,lambda_reg)
%% initial
if nargin < 7
    lambda_reg = 0; % 默认不加正则化，向后兼容
end

num_image=size(normalized_Image,2);
for i=1:1:num_image
    point_image_num(i)=size(normalized_Image{i}.point_2d,2);
end

% 参数总长度
total_vars = sum(point_image_num)*2;

% 零初始化
x0 = zeros(total_vars, 1);

order=[];
for i=1:1:size(point,2)
    if isempty(point{i})==0
        for j=1:1:size(point{i},2)
            order=[order,2*sum(point_image_num(1:i-1))+2*point{i}(j)-1,2*sum(point_image_num(1:i-1))+2*point{i}(j)];
        end
    end
end
x0(order)=[];

%% Solve using lsqnonlin (LM / trust region)
opts = optimoptions(@lsqnonlin,'SpecifyObjectiveGradient',true,'MaxIterations',40,'Display','iter');
[X_update_k,resnorm,res,eflag,output2] = lsqnonlin(@(x)myfun1(x,J,normalized_Image,measurements,order,lambda_reg,point_image_num),x0,[],[],opts);
for i=1:1:size(order,2)
    X_update_k=[X_update_k(1:1:order(i)-1);0;X_update_k(order(i):end)];
end

end
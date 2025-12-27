function [X_update_k]=LLS(J,normalized_Image,measurements,k1_all,k2_all,point)
%% inital
num_image=size(normalized_Image,2);%Image size
for i=1:1:num_image
    point_image_num(i)=size(normalized_Image{i}.point_2d,2);
end
% X_update_k=zeros(sum(point_image_num)*2,1);
k_one=[];
k_two=[];
for i=1:1:size(k1_all,1)
    k_one=[k_one,k1_all(i,:)];
    k_two=[k_two,k2_all(i,:)];
end
X_update_k(1:2:size(k_one,2)*2-1,1)=k_one';
X_update_k(2:2:size(k_one,2)*2,1)=k_two';
if sum(point_image_num)*2~=size(X_update_k,1)
    x0=-zeros(sum(point_image_num)*2,1);
else
    x0 = X_update_k; % Starting guess
end
order=[];
for i=1:1:size(point,2)
    if isempty(point{i})==0
        for j=1:1:size(point{i},2)
            order=[order,2*sum(point_image_num(1:i-1))+2*point{i}(j)-1,2*sum(point_image_num(1:i-1))+2*point{i}(j)];
        end
    end
end
x0(order)=[];
%% upldate
algorithm=0; % 1 means the GN method, 0 using the tool box
if algorithm
    for k=1:1:15 %converge generation
        %% Special Jaccobi matrix
        [FX,Jk]=myfun(x0,J,normalized_Image,measurements);
        Z=zeros(size(FX,1),1);
        %% Guasss Newton
        Jk=sparse(Jk);
    %     X_update_k_1=inv(Jk'*inv(Pw)*Jk)*Jk'*inv(Pw)*(Z-FX+Jk*X_update_k);
        X_update_k_1=(Jk'*Jk)\Jk'*(Z-FX+Jk*x0);
        disp(norm(X_update_k_1-x0));
        if norm(X_update_k_1-x0)<1
            disp(k);
    %         for i=1:1:it
    %             if X_update_k_1(3*i)>pi
    %                 X_update_k_1(3*i)=X_update_k_1(3*i)-2*pi;
    %             else
    %                 X_update_k_1(3*i)=X_update_k_1(3*i)+2*pi;
    %             end
    %         end
    %         PX=inv(Jk'*inv(Pw)*Jk);
            X_update_k=X_update_k_1;
            break;
        end
        x0=X_update_k_1;
    end
else
    %% Solve nonlinear least-squares (nonlinear data-fitting) problems (MATLAB, LM, trust region)
    opts = optimoptions(@lsqnonlin,'SpecifyObjectiveGradient',true,'MaxIterations',40);%,'CheckGradients',true
    % opts.Algorithm = 'levenberg-marquardt';
    [X_update_k,resnorm,res,eflag,output2] = lsqnonlin(@(x)myfun1(x,J,normalized_Image,measurements,order),x0,[],[],opts);
    for i=1:1:size(order,2)
        X_update_k=[X_update_k(1:1:order(i)-1);0;X_update_k(order(i):end)];
    end
end


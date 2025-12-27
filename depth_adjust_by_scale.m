function [depth]=depth_adjust_by_scale(depth,value)
%depth_scale=fix(mean(mean(fix(value./depth))));
%depth=depth_scale.*depth;

A=fix(value./depth);
for i=1:1:size(A,1)
    depth_scale(i)=fix(mean(mean(A(i,find(A(i,:)~=inf)))));
end
depth_scale_1=mean(depth_scale);
depth=depth_scale_1.*depth;


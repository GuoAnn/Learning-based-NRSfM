function [ep1]=draw_image_dense(Shape_A, Shape_B)
    %% Ground truth
    P=Shape_A;
    P1o=Shape_B;
    [d,P1_old,transform] = procrustes(P,P1o);
    idx = find(P1o(:,1)~=0);
    idx1=[];
    for j=1:1:size(P,1)
        if norm(P1_old(j,:)-P(j,:))<50
            idx1=[idx1,j];
        end
    end
    idx2 = intersect(idx,idx1);
    [d,P11,transform] = procrustes(P(idx1,:),P1o(idx1,:));
    P1x(idx2,:)=P11;
    sc= sqrt(sum(sum(P(idx2,:).^2)));
    ep1 = sqrt(sum(sum((P1x(idx2,:)-P(idx2,:)).^2)))/sc;
    figure,
    plot3(P1x(:,1),P1x(:,2),P1x(:,3),'*g');
    hold on
    plot3(P(:,1),P(:,2),P(:,3),'*b');
end
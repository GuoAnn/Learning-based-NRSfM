function [ep1, P2, scale]=draw_image_sparse(Shape_A, Shape_B)
    Pgth =  Shape_B;%scene.Pgth(i).P;
    qw = Shape_A;
    %idx2 = Pgth(2*(i-1)+1,:)~=0;
    [~,qw,~]= absor(qw,Pgth,'doScale',true);
    P2 = qw;
    scale = max(max(Pgth')-min(Pgth'));
    ep1 = mean(sqrt(mean((Pgth-P2).^2))/scale);
    figure,
    plot3(Pgth(1,:),Pgth(2,:),Pgth(3,:),'*g');
    hold on
    plot3(P2(1,:),P2(2,:),P2(3,:),'*b');
end
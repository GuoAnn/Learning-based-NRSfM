function [ep1, P2, scale]=draw_image_sparse_with_image(Shape_A, Shape_B)
    
    persistent frame_id;
    if isempty(frame_id), frame_id = 0; end
    frame_id = frame_id + 1;

    Pgth =  Shape_B;%scene.Pgth(i).P;
    qw = Shape_A;
    %idx2 = Pgth(2*(i-1)+1,:)~=0;
    [~,qw,~]= absor(qw,Pgth,'doScale',true); % 相似变换对齐
    P2 = qw;
    scale = max(max(Pgth')-min(Pgth')); % 归一化尺度
    ep1 = mean(sqrt(mean((Pgth-P2).^2))/scale);
    figure,
    plot3(Pgth(1,:),Pgth(2,:),Pgth(3,:),'*g');hold on;
    plot3(P2(1,:),P2(2,:),P2(3,:),'*b');

    axis equal; grid on;
    fname = sprintf('reconstruction_%04d.png', frame_id);
    saveas(gcf, fname);% 每次调用保存不同文件名保存一张图片,每帧一张
end


% function [ep1, P2, scale] = draw_image_sparse(Shape_A, Shape_B)
%     Pgth = Shape_B;
%     qw = Shape_A;
% 
%     [~, qw, ~] = absor(qw, Pgth, 'doScale', true);
%     P2 = qw;
%     scale = max(max(Pgth') - min(Pgth'));
%     ep1 = mean(sqrt(mean((Pgth - P2).^2)) / scale);
% 
%     figure;
%     % Ground Truth用绿色星号
%     plot3(Pgth(1,:), Pgth(2,:), Pgth(3,:), 'g*', 'MarkerSize', 8, 'LineWidth', 1.5);
%     hold on;
%     % Reconstruction用蓝色圆圈
%     plot3(P2(1,:), P2(2,:), P2(3,:), 'bo', 'MarkerSize', 6, 'LineWidth', 1);
% 
%     title(sprintf('3D Shape Alignment (Error: %.4f)', ep1));
%     legend('Ground Truth', 'Reconstruction', 'Location', 'best');
%     grid on; axis equal;
%     xlabel('X'); ylabel('Y'); zlabel('Z');
%     view(45, 30);  % 更好的视角
% 
%     % 添加误差文本
%     text(0.05, 0.95, sprintf('Error: %.4f', ep1), ...
%          'Units', 'normalized', 'FontSize', 11, ...
%          'BackgroundColor', 'white');
% end
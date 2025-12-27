function [F,Jk] = myfun(x,J,normalized_Image,measurements,order)
for i=1:1:size(order,2)
    x=[x(1:1:order(i)-1);0;x(order(i):end)];
end
num_measure=size(measurements,2);%Measure image size
num_image=size(normalized_Image,2);%Image size
for i=1:1:num_measure
    point_num(i)=size(measurements{i}.point,1);
end
for i=1:1:num_image
    point_image_num(i)=size(normalized_Image{i}.point_2d,2);
end
row=[];
col=[];
val=[];
for i=1:1:num_measure
    for j=1:1:size(measurements{i}.point,1)
%         measure_k=sum(point_num(1:i-1))+measurements{i}.point(j,1);
%         measure_l=sum(point_num(1:i-1))+measurements{i}.point(j,2);
        img_in=measurements{i}.image(1);
        img_out=measurements{i}.image(2);
        img_in_point=sum(point_image_num(1:img_in-1))+measurements{i}.point(j,1);
        img_out_point=sum(point_image_num(1:img_out-1))+measurements{i}.point(j,2);
        k1=x(2*img_in_point-1);
        k2=x(2*img_in_point);
        k1_bar=x(2*img_out_point-1);
        k2_bar=x(2*img_out_point);
        x1=normalized_Image{img_in}.point_2d(1,measurements{i}.point(j,1));
        x2=normalized_Image{img_in}.point_2d(2,measurements{i}.point(j,1));
        x1_bar=normalized_Image{img_out}.point_2d(1,measurements{i}.point(j,2));
        x2_bar=normalized_Image{img_out}.point_2d(2,measurements{i}.point(j,2));
        dx1_dy1=J.dx1_dy1(i,measurements{i}.point(j,2));
        dx1_dy2=J.dx1_dy2(i,measurements{i}.point(j,2));
        dx2_dy1=J.dx2_dy1(i,measurements{i}.point(j,2));
        dx2_dy2=J.dx2_dy2(i,measurements{i}.point(j,2));
        dy1_dx1=dx2_dy2/(dx1_dy1*dx2_dy2-dx1_dy2*dx2_dy1);
        dy1_dx2=-dx1_dy2/(dx1_dy1*dx2_dy2-dx1_dy2*dx2_dy1);
        dy2_dx1=-dx2_dy1/(dx1_dy1*dx2_dy2-dx1_dy2*dx2_dy1);
        dy2_dx2=dx1_dy1/(dx1_dy1*dx2_dy2-dx1_dy2*dx2_dy1);
        ddx1_dy1dy2=J.ddx1_dxdy(i,measurements{i}.point(j,2));
        ddx2_dy1dy2=J.ddx2_dxdy(i,measurements{i}.point(j,2));
        ddx1_dy1dy1=J.ddx1_ddy1(i,measurements{i}.point(j,2));
        ddx2_dy1dy1=J.ddx2_ddy1(i,measurements{i}.point(j,2));
        e1=x1^2+x2^2+1;
        e1_bar=x1_bar^2+x2_bar^2+1;
        b1=x1^2+1;
        b1_bar=x1_bar^2+1;
        b2=x2^2+1;
        b2_bar=x2_bar^2+1;
        g11=(k1*x1-1)^2+b2*k1^2;
        g12=e1*k1*k2-k2*x1-k1*x2;
        g21=g12;
        g22=(k2*x2-1)^2+b1*k2^2;
        g11_bar=(k1_bar*x1_bar-1)^2+b2_bar*k1_bar^2;
        g12_bar=e1_bar*k1_bar*k2_bar-k2_bar*x1_bar-k1_bar*x2_bar;
        g21_bar=g12_bar;
        g22_bar=(k2_bar*x2_bar-1)^2+b1_bar*k2_bar^2;
        %% Derivative %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        df1_dk1=-1*dx1_dy1;
        df1_dk2=-1*dx2_dy1;
        df1_dk1_bar=1*1;
        df1_dk2_bar=0;
        df2_dk1=-1*dx1_dy2;
        df2_dk2=-1*dx2_dy2;
        df2_dk1_bar=0;
        df2_dk2_bar=1*1;
        df3_dk1=dx1_dy1*dx1_dy1*(-2*e1*k1*x1_bar*k2_bar + 2*e1*k1*e1_bar*k1_bar*k2_bar-2*e1*k1*x2_bar*k1_bar + 2*x1*x1_bar*k2_bar- 2*x1*e1_bar*k1_bar*k2_bar + 2*x1*x2_bar*k1_bar)+2*dx1_dy1*dx2_dy1*(-e1*x1_bar*k2*k2_bar+e1*e1_bar*k2*k1_bar*k2_bar-e1*x2_bar*k2*k1_bar+x2*k2_bar*x1_bar-x2*e1_bar*k1_bar*k2_bar+x2*x2_bar*k1_bar)+...
                dx2_dy1*dx2_dy1*0-dx1_dy1*dx1_dy2*(2*k1*x1^2*k1_bar^2*x1_bar^2-4*k1*x1^2*x1_bar*k1_bar+2*k1*x1^2+2*k1*x1^2*b2_bar*k1_bar^2-2*x1*x1_bar^2*k1_bar^2+4*x1*x1_bar*k1_bar-2*x1-2*x1*b2_bar*k1_bar^2+2*b2*k1*x1_bar^2*k1_bar^2-4*b2*k1*x1_bar*k1_bar+2*b2*k1+2*b2*k1*b2_bar*k1_bar^2)...
                -(dx1_dy1*dx2_dy2+dx2_dy1*dx1_dy2)*(e1*k2*k1_bar^2*x1_bar^2-2*e1*k2*k1_bar*x1_bar+e1*k2+e1*k2*b2_bar*k1_bar^2-x2*k1_bar^2*x1_bar^2+2*x2*k1_bar*x1_bar-x2-x2*b2_bar*k1_bar^2)-dx2_dy1*dx2_dy2*0;
        df3_dk2=dx1_dy1*dx1_dy1*0+2*dx2_dy1*dx1_dy1*(k2_bar*x1*x1_bar-e1_bar*x1*k1_bar*k2_bar+x1*x2_bar*k1_bar-e1*x1_bar*k1*k2_bar+e1*e1_bar*k1*k1_bar*k2_bar-e1*x2_bar*k1*k1_bar)+...
                dx2_dy1*dx2_dy1*(-2*x1_bar*x2^2*k2_bar*k2+2*x2^2*e1_bar*k2*k1_bar*k2_bar-2*x2^2*x2_bar*k2*k1_bar+2*x2*x1_bar*k2_bar-2*x2*e1_bar*k1_bar*k2_bar+2*x2*x2_bar*k1_bar-2*b1*k2*x1_bar*k2_bar+2*b1*e1_bar*k2*k1_bar*k2_bar-2*b1*x2_bar*k2*k1_bar)-dx1_dy1*dx1_dy2*0-...
                (dx1_dy1*dx2_dy2+dx2_dy1*dx1_dy2)*(-x1*k1_bar^2*x1_bar^2+2*x1*k1_bar*x1_bar-x1-x1*b2_bar*k1_bar^2+e1*k1*k1_bar^2*x1_bar^2-2*e1*k1*k1_bar*x1_bar+e1*k1+e1*k1*b2_bar*k1_bar^2)-...
                dx2_dy1*dx2_dy2*(2*x2^2*k2*k1_bar^2*x1_bar^2-4*x2^2*k2*k1_bar*x1_bar+2*x2^2*k2+2*x2^2*k2*b2_bar*k1_bar^2-2*x2*k1_bar^2*x1_bar^2+4*x2*k1_bar*x1_bar-2*x2-2*x2*b2_bar*k1_bar^2+2*b1*k2*k1_bar^2*x1_bar^2-4*b1*k2*k1_bar*x1_bar+2*b1*k2+2*b1*k2*b2_bar*k1_bar^2);
        df3_dk1_bar=dx1_dy1*dx1_dy1*(e1*k1^2*e1_bar*k2_bar-e1*k1^2*x2_bar- 2*k1*x1*e1_bar*k2_bar + 2*k1*x1*x2_bar + e1_bar*k2_bar - x2_bar)+2*dx1_dy1*dx2_dy1*(-e1_bar*x1*k2*k2_bar+x1*x2_bar*k2+e1*e1_bar*k1*k2*k2_bar-e1*x2_bar*k1*k2-x2*k1*e1_bar*k2_bar+x2*k1*x2_bar)+...
                dx2_dy1*dx2_dy1*(x2^2*e1_bar*k2^2*k2_bar-x2^2*x2_bar*k2^2-2*x2*k2*e1_bar*k2_bar+2*x2*k2*x2_bar+e1_bar*k2_bar-x2_bar+b1*e1_bar*k2^2*k2_bar-b1*x2_bar*k2^2)-dx1_dy1*dx1_dy2*(2*k1^2*x1^2*k1_bar*x1_bar^2-2*k1^2*x1^2*x1_bar+2*k1^2*x1^2*b2_bar*k1_bar-4*k1*x1*x1_bar^2*k1_bar+4*k1*x1*x1_bar-4*k1*x1*b2_bar*k1_bar+2*x1_bar^2*k1_bar-2*x1_bar+2*b2_bar*k1_bar+2*b2*k1^2*x1_bar^2*k1_bar-2*b2*k1^2*x1_bar+2*b2*k1^2*b2_bar*k1_bar)...
                -(dx1_dy1*dx2_dy2+dx2_dy1*dx1_dy2)*(-2*x1*k2*k1_bar*x1_bar^2+2*x1*k2*x1_bar-2*x1*k2*b2_bar*k1_bar+2*e1*k1*k2*k1_bar*x1_bar^2-2*e1*k1*k2*x1_bar+2*e1*k1*k2*b2_bar*k1_bar-2*x2*k1*k1_bar*x1_bar^2+2*x2*k1*x1_bar-2*x2*k1*b2_bar*k1_bar)-dx2_dy1*dx2_dy2*(2*x2^2*k2^2*k1_bar*x1_bar^2-2*x2^2*k2^2*x1_bar+2*x2^2*k2^2*b2_bar*k1_bar-4*x2*k2*k1_bar*x1_bar^2+4*x2*k2*x1_bar-4*x2*k2*b2_bar*k1_bar+2*k1_bar*x1_bar^2-2*x1_bar+2*b2_bar*k1_bar+2*b1*k2^2*k1_bar*x1_bar^2-2*b1*k2^2*x1_bar+2*b1*k2^2*b2_bar*k1_bar);
        df3_dk2_bar=dx1_dy1*dx1_dy1*(-e1*k1^2*x1_bar + e1*k1^2*e1_bar*k1_bar + 2*k1*x1*x1_bar- 2*k1*x1*e1_bar*k1_bar -x1_bar + e1_bar*k1_bar)+2*dx1_dy1*dx2_dy1*(k2*x1*x1_bar-e1_bar*x1*k2*k1_bar-e1*x1_bar*k1*k2+e1*e1_bar*k1*k2*k1_bar+x2*k1*x1_bar-x2*k1*e1_bar*k1_bar)...
                +dx2_dy1*dx2_dy1*(-x1_bar*x2^2*k2^2+x2^2*e1_bar*k2^2*k1_bar+2*x2*x1_bar*k2-2*x2*k2*e1_bar*k1_bar-x1_bar+e1_bar*k1_bar-b1*k2^2*x1_bar+b1*e1_bar*k2^2*k1_bar)-dx1_dy1*dx1_dy2*0-dx1_dy1*dx2_dy2*0-dx2_dy1*dx1_dy2*0-dx2_dy1*dx2_dy2*0;
        df4_dk1=dx1_dy1*dx1_dy1*(2*e1*k1*x2_bar^2*k2_bar^2-4*e1*k1*x2_bar*k2_bar+2*e1*k1+2*e1*k1*b1_bar*k2_bar^2-2*x1*x2_bar^2*k2_bar^2+4*x1*x2_bar*k2_bar-2*x1-2*x1*b1_bar*k2_bar^2)+2*dx1_dy1*dx2_dy1*(e1*k2*x2_bar^2*k2_bar^2-2*e1*k2*x2_bar*k2_bar+e1*k2+e1*k2*b1_bar*k2_bar^2-x2*x2_bar^2*k2_bar^2+2*x2*x2_bar*k2_bar-x2-x2*b1_bar*k2_bar^2)+dx2_dy1*dx2_dy1*0....
                -dx1_dy2*dx1_dy2*(2*k1*x1^2*k1_bar^2*x1_bar^2-4*k1*x1^2*x1_bar*k1_bar+2*k1*x1^2+2*k1*x1^2*b2_bar*k1_bar^2-2*x1*x1_bar^2*k1_bar^2+4*x1*x1_bar*k1_bar-2*x1-2*x1*b2_bar*k1_bar^2+2*b2*k1*x1_bar^2*k1_bar^2-4*b2*k1*x1_bar*k1_bar+2*b2*k1+2*b2*k1*b2_bar*k1_bar^2)...
                -(dx1_dy2*dx2_dy2+dx2_dy2*dx1_dy2)*(e1*k2*k1_bar^2*x1_bar^2-2*e1*k2*k1_bar*x1_bar+e1*k2+e1*k2*b2_bar*k1_bar^2-x2*k1_bar^2*x1_bar^2+2*x2*k1_bar*x1_bar-x2-x2*b2_bar*k1_bar^2)-dx2_dy2*dx2_dy2*0;
        df4_dk2=dx1_dy1*dx1_dy1*0+2*dx1_dy1*dx2_dy1*(-x1*x2_bar^2*k2_bar^2+2*x1*x2_bar*k2_bar-x1-x1*b1_bar*k2_bar^2+e1*k1*x2_bar^2*k2_bar^2-2*e1*k1*x2_bar*k2_bar+e1*k1+e1*k1*b1_bar*k2_bar^2)...
                +dx2_dy1*dx2_dy1*(2*x2^2*k2*x2_bar^2*k2_bar^2-4*x2^2*k2*x2_bar*k2_bar+2*x2^2*k2+2*x2^2*k2*b1_bar*k2_bar^2-2*x2*k2_bar^2*x2_bar^2+4*x2*k2_bar*x2_bar-2*x2-2*x2*b1_bar*k2_bar^2+2*b1*k2*x2_bar^2*k2_bar^2-4*b1*k2*x2_bar*k2_bar+2*b1*k2+2*b1*k2*b1_bar*k2_bar^2)...
                -dx1_dy2*dx1_dy2*0-...
                (dx1_dy2*dx2_dy2+dx2_dy2*dx1_dy2)*(-x1*k1_bar^2*x1_bar^2+2*x1*k1_bar*x1_bar-x1-x1*b2_bar*k1_bar^2+e1*k1*k1_bar^2*x1_bar^2-2*e1*k1*k1_bar*x1_bar+e1*k1+e1*k1*b2_bar*k1_bar^2)-...
                dx2_dy2*dx2_dy2*(2*x2^2*k2*k1_bar^2*x1_bar^2-4*x2^2*k2*k1_bar*x1_bar+2*x2^2*k2+2*x2^2*k2*b2_bar*k1_bar^2-2*x2*k1_bar^2*x1_bar^2+4*x2*k1_bar*x1_bar-2*x2-2*x2*b2_bar*k1_bar^2+2*b1*k2*k1_bar^2*x1_bar^2-4*b1*k2*k1_bar*x1_bar+2*b1*k2+2*b1*k2*b2_bar*k1_bar^2);
        df4_dk1_bar=dx1_dy1*dx1_dy1*0+2*dx1_dy1*dx2_dy1*0+dx2_dy1*dx2_dy1*0-dx1_dy2*dx1_dy2*(2*k1^2*x1^2*k1_bar*x1_bar^2-2*k1^2*x1^2*x1_bar+2*k1^2*x1^2*b2_bar*k1_bar-4*k1*x1*x1_bar^2*k1_bar+4*k1*x1*x1_bar-4*k1*x1*b2_bar*k1_bar+2*x1_bar^2*k1_bar-2*x1_bar+2*b2_bar*k1_bar+2*b2*k1^2*x1_bar^2*k1_bar-2*b2*k1^2*x1_bar+2*b2*k1^2*b2_bar*k1_bar)...
                -(dx1_dy2*dx2_dy2+dx2_dy2*dx1_dy2)*(-2*x1*k2*k1_bar*x1_bar^2+2*x1*k2*x1_bar-2*x1*k2*b2_bar*k1_bar+2*e1*k1*k2*k1_bar*x1_bar^2-2*e1*k1*k2*x1_bar+2*e1*k1*k2*b2_bar*k1_bar-2*x2*k1*k1_bar*x1_bar^2+2*x2*k1*x1_bar-2*x2*k1*b2_bar*k1_bar)-dx2_dy2*dx2_dy2*(2*x2^2*k2^2*k1_bar*x1_bar^2-2*x2^2*k2^2*x1_bar+2*x2^2*k2^2*b2_bar*k1_bar-4*x2*k2*k1_bar*x1_bar^2+4*x2*k2*x1_bar-4*x2*k2*b2_bar*k1_bar+2*k1_bar*x1_bar^2-2*x1_bar+2*b2_bar*k1_bar+2*b1*k2^2*k1_bar*x1_bar^2-2*b1*k2^2*x1_bar+2*b1*k2^2*b2_bar*k1_bar);
        df4_dk2_bar=dx1_dy1*dx1_dy1*(2*e1*k1^2*x2_bar^2*k2_bar-2*e1*k1^2*x2_bar+2*e1*k1^2*b1_bar*k2_bar-4*k1*x1*x2_bar^2*k2_bar+4*k1*x1*x2_bar-4*k1*x1*b1_bar*k2_bar+2*x2_bar^2*k2_bar-2*x2_bar+2*b1_bar*k2_bar)+2*dx1_dy1*dx2_dy1*(-2*x1*k2*x2_bar^2*k2_bar+2*x1*k2*x2_bar-2*x1*k2*b1_bar*k2_bar+2*e1*k1*k2*x2_bar^2*k2_bar-2*e1*k1*k2*x2_bar+2*e1*k1*k2*b1_bar*k2_bar-2*x2*k1*x2_bar^2*k2_bar+2*x2*k1*x2_bar-2*x2*k1*b1_bar*k2_bar)+dx2_dy1*dx2_dy1*(2*x2^2*k2^2*x2_bar^2*k2_bar-2*x2^2*k2^2*x2_bar+2*x2^2*k2^2*b1_bar*k2_bar-4*x2*k2*k2_bar*x2_bar^2+4*x2*k2*x2_bar-4*x2*k2*b1_bar*k2_bar+2*x2_bar^2*k2_bar-2*x2_bar+2*b1_bar*k2_bar+2*b1*k2^2*x2_bar^2*k2_bar-2*b1*k2^2*x2_bar+2*b1*k2^2*b1_bar*k2_bar)...
                -dx1_dy2*dx1_dy2*0-dx1_dy2*dx2_dy2*0-dx2_dy2*dx1_dy2*0-dx2_dy2*dx2_dy2*0;        
%         Jk(sum(point_num(1:i-1))*4+j*4-3:sum(point_num(1:i-1))*4+j*4,img_in_point*2-1:2*img_in_point)=[df1_dk1,df1_dk2;
%                                                                                  df2_dk1,df2_dk2;
%                                                                                  df3_dk1,df3_dk2;
%                                                                                  df4_dk1,df4_dk2;];
%         Jk(sum(point_num(1:i-1))*4+j*4-3:sum(point_num(1:i-1))*4+j*4,img_out_point*2-1:2*img_out_point)=[df1_dk1_bar,df1_dk2_bar;
%                                                                                  df2_dk1_bar,df2_dk2_bar;
%                                                                                  df3_dk1_bar,df3_dk2_bar;
%                                                                                  df4_dk1_bar,df4_dk2_bar;];
        a=sum(point_num(1:i-1))*4+j*4;
        b=img_in_point*2;
        c=2*img_out_point;
        row=[row,a-3,a-3,a-2,a-2,a-1,a-1,a,a,a-3,a-3,a-2,a-2,a-1,a-1,a,a];
        col=[col,b-1,b,b-1,b,b-1,b,b-1,b,c-1,c,c-1,c,c-1,c,c-1,c];
        val=[val,df1_dk1,df1_dk2,df2_dk1,df2_dk2,df3_dk1,df3_dk2,df4_dk1,df4_dk2,df1_dk1_bar,df1_dk2_bar,df2_dk1_bar,df2_dk2_bar,df3_dk1_bar,df3_dk2_bar,df4_dk1_bar,df4_dk2_bar;];
        %% Function
        F(sum(point_num(1:i-1))*4+j*4-3:sum(point_num(1:i-1))*4+j*4,1)=[1*(k1_bar-dx1_dy1*k1-dx2_dy1*k2+dy2_dx1*ddx1_dy1dy2+dy2_dx2*ddx2_dy1dy2);%dy1_dx1*ddx1_dy1dy1/2+dy1_dx2*ddx2_dy1dy1/2;
                                                                         1*(k2_bar-dx1_dy2*k1-dx2_dy2*k2+dy1_dx1*ddx1_dy1dy2+dy1_dx2*ddx2_dy1dy2);
                                                                         dx1_dy1*dx1_dy1*g11*g12_bar+dx1_dy1*dx2_dy1*g12*g12_bar+dx2_dy1*dx1_dy1*g21*g12_bar+dx2_dy1*dx2_dy1*g22*g12_bar-dx1_dy1*dx1_dy2*g11*g11_bar-dx1_dy1*dx2_dy2*g12*g11_bar-dx2_dy1*dx1_dy2*g21*g11_bar-dx2_dy1*dx2_dy2*g22*g11_bar;
                                                                         dx1_dy1*dx1_dy1*g11*g22_bar+dx1_dy1*dx2_dy1*g12*g22_bar+dx2_dy1*dx1_dy1*g21*g22_bar+dx2_dy1*dx2_dy1*g22*g22_bar-dx1_dy2*dx1_dy2*g11*g11_bar-dx1_dy2*dx2_dy2*g12*g11_bar-dx2_dy2*dx1_dy2*g21*g11_bar-dx2_dy2*dx2_dy2*g22*g11_bar;];
    end
end
% Jk=sparse(Jk);
Jk=sparse(row,col,val,sum(point_num)*4,size(x,1));
% if max(max(abs(Jk-Jk1)))>1e-6
%     aa=1;
% end
Jk(:,order)=[];
end
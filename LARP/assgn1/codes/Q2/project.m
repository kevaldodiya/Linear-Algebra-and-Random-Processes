function [A1] = project(P1,sigma,P2,n,k)
A1 = zeros(size(P1,1),size(P2,1));
number =0;
if size(P1,1)<size(P2,1)
    number = size(P1,1).*(n/100);
    number = int64(number);
else
    number = size(P2,1).*(n/100);
    number = int64(number);
end
if k == 0
    temp =0;
    if size(P1,1)<size(P2,1)
        temp = randperm(size(P1,1),number);
    else
        temp = randperm(size(P2,1),number);
    end
    for i=1:size(temp,2)
        j = temp(i);
        A1=A1+sigma(j,j)*P1(:,j)*transpose(P2(:,j));
    end
else
    for i=1:number
        A1=A1+sigma(i,i)*P1(:,i)*transpose(P2(:,i));
    end
end
end
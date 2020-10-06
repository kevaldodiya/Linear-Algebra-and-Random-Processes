function [P,D] = setEigen(P,D)
ar = [];
for i = 1:size(D,1)
    ar = [ar ,real(D(i,i))];
end
[sorted,index]=sort(ar);
for i = 1:size(ar,2)
    D(i,i) =sorted((size(ar,2) -i)+1);
end
P1 = zeros(size(D,1));
for i = 1:size(ar,2)
    P1(:,i) = P(:,index((size(ar,2)-i)+1));
end
P = P1;
    

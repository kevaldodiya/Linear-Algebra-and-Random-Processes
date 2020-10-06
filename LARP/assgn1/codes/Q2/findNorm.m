function [norm] = findNorm(A)
sum =0;   
for i=1:size(A,1)
        for j=1:size(A,2)
          sum = sum + A(i,j).* A(i,j);
        end
end
norm = sqrt(sum);
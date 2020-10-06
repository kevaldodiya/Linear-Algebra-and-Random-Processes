function [P] = MakeNormal(P1)
sum = 0;
P =[];
for j=1:size(P1,2)
        for i=1:size(P1,1)
            sum = sum + P1(i,j).* P1(i,j);
        end
      sum = sqrt(sum);
      temp = (1/sum)*P1(:,j);
      P = [P,temp];
end
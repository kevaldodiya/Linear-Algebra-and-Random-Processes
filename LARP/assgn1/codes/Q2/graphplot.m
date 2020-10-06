function [k] = graphplot(P1,sigma,P2,n2)
k = [];
j = [];
for i=10:5:90
    j = [j;i];
    A1=project(P1,sigma,P2,i,1);
    n1 = findNorm(A1);
    e = n2 - n1;
    k = [k;e];
end
figure(3)
plot(j,k);
xlabel(" N -->");
ylabel("Frobenius norm diff(error) -->");
title("N vs error");
end
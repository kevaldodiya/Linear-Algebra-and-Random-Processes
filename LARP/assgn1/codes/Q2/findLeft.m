function [P2] = findLeft(A,P1,D1,n,m)
 if (n>m)
     A_Atrans = A*transpose(A);
     [P2,D2] = eig(A_Atrans);
     [P2,D2] = setEigen(P2,D2);
     P2 = MakeNormal(P2);
 else
     P2 = [];
        for i=1:n
            temp = (1/D1(i,i))*A*P1(:,i);
            P2 = [P2,temp];
        end
 end
end
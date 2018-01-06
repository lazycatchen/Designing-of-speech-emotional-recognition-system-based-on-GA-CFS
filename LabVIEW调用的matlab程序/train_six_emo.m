
function [nan_rank,testt,input_train,w1,b1,w2,b2]=train_six_emo(x1,x2,x3,x4,x5,x6,n)
happy=x1;
angry=x2;
sad=x3;
neutral=x4;
surprise=x5;
fear=x6;

train_number=n;
data=[happy;angry;sad;neutral;surprise;fear]';
nan_rank=find(any(isnan(data),2)==1);
data(any(isnan(data),2),:)=[];

data=data';
H=ttest2(data(1:50,:),data(101:150,:));
testt=find(H==1);
data1=data(:,testt);
add1=ones(size(happy,1),1);
add2=ones(size(angry,1),1)*2;
add3=ones(size(sad,1),1)*3;
add4=ones(size(neutral,1),1)*4;
add5=ones(size(surprise,1),1)*5;
add6=ones(size(fear,1),1)*6;
add=[add1;add2;add3;add4;add5;add6];
data=[add data1];
all_row=size(data1,1);
all_rank= size(data,2);
train_row=all_row*1;
k=rand(1,all_row);
[m,nn]=sort(k);    input=data(:,2:all_rank);
output1 =data(:,1);
for i=1:all_row
       switch output1(i)
        case 1
            output(i,:)=[1 0 0 0 0 0];
        case 2
            output(i,:)=[0 1 0 0 0 0];
        case 3
            output(i,:)=[0 0 1 0 0 0];
        case 4
            output(i,:)=[0 0 0 1 0 0];
        case 5
            output(i,:)=[0 0 0 0 1 0];
        case 6
            output(i,:)=[0 0 0 0 0 1];
       end
end
input_train=input(nn(1:train_row),:)';
output_train=output(nn(1:train_row),:)';
[inputn,inputps]=mapminmax(input_train);
innum=all_rank-1;midnum=(all_rank);
outnum=6;
w1=rands(midnum,innum);
b1=rands(midnum,1);
w2=rands(midnum,outnum);
b2=rands(outnum,1);

w2_1=w2;w2_2=w2_1;
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

%学习率
xite=0.1;
alfa=0.01;
for ii=1:train_number
   
    E(ii)=0;
    for i=1:1:train_row
       %% 网络预测输出 
        x=inputn(:,i);
        % 隐含层输出
        for j=1:1:midnum
            I(j)=inputn(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        % 输出层输出
        yn=w2'*Iout'+b2;
        
       %% 权值阀值修正
        %计算误差
        e=output_train(:,i)-yn;     
        E(ii)=E(ii)+sum(abs(e));
        
        %计算权值变化率
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j)));
            FI(j)=S*(1-S);
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
            end
        end
           
        w1=w1_1+xite*dw1'+alfa*(w1_1-w1_2);
        b1=b1_1+xite*db1'+alfa*(b1_1-b1_2);
        w2=w2_1+xite*dw2'+alfa*(w2_1-w2_2);
        b2=b2_1+xite*db2'+alfa*(b2_1-b2_2);
        
        w1_2=w1_1;w1_1=w1;
        w2_2=w2_1;w2_1=w2;
        b1_2=b1_1;b1_1=b1;
        b2_2=b2_1;b2_1=b2;
    end

end


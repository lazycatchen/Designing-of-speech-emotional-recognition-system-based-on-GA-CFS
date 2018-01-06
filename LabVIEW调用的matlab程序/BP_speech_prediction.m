clear
clc;
happy=importdata('E:\aa0011\音乐+脑电\database\casia汉语情感语料库\相同文本50\liuchanhg\datasum_happy.mat');
angry=importdata('E:\aa0011\音乐+脑电\database\casia汉语情感语料库\相同文本50\liuchanhg\datasum_angry.mat');
surprise=importdata('E:\aa0011\音乐+脑电\database\casia汉语情感语料库\相同文本50\liuchanhg\datasum_surprise.mat');
sad=importdata('E:\aa0011\音乐+脑电\database\casia汉语情感语料库\相同文本50\liuchanhg\datasum_sad.mat');
fear=importdata('E:\aa0011\音乐+脑电\database\casia汉语情感语料库\相同文本50\liuchanhg\datasum_fear.mat');
neutral=importdata('E:\aa0011\音乐+脑电\database\casia汉语情感语料库\相同文本50\liuchanhg\datasum_neutral.mat');
% data=[happy(:,1:60);angry(:,1:60);surprise(:,1:60);sad(:,1:60);fear(:,1:60);neutral(:,1:60)];
data=[happy;angry;surprise;sad;fear;neutral]';
% lable1=ones(1,50)';
% lable=[lable1;lable1*2;lable1*3;lable1*4;lable1*5;lable1*6];
% data=[lable data];
% 
% data(isnan(data))=0;
data(any(isnan(data),2),:)=[];
data=data';
data1=data;
% [H,P,CI]=ttest2(data(1:50,:),data(151:200,:));
% testt=find(H==1);
% data1=data(:,testt);

aaa= size(data1,2);
add1=ones(size(data,1)/6,1);
add2=ones(size(data,1)/6,1)*2;
add3=ones(size(data,1)/6,1)*3;
add4=ones(size(data,1)/6,1)*4;
add5=ones(size(data,1)/6,1)*5;
add6=ones(size(data,1)/6,1)*6;
% add7=ones(size(data,1)/6,1)*7;
add=[add1;add2;add3;add4;add5;add6];%%数据标号
data=[add data1];

%从1到2000间随机排序
k=rand(1,300);
[m,n]=sort(k);

%输入输出数据
input=data(:,2:aaa);
output1 =data(:,1);
for i=1:300
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
%         case 7
%             output(i,:)=[0 0 0 0 0 0 1];
    end
end


%随机提取1500个样本为训练样本，500个样本为预测样本
input_train=input(n(1:270),:)';
output_train=output(n(1:270),:)';
% input_test=input(n(271:300),:)';
% output_test=output(n(271:300),:)';
%输入数据归一化
[inputn,inputps]=mapminmax(input_train);


%% 网络结构初始化
innum=aaa-1;
midnum=(aaa);
outnum=6;
 

%权值初始化
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

%% 网络训练
for ii=1:10
   
    E(ii)=0;
    for i=1:1:270
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

%% 语音特征信号分类
input_test=input(n(272),:)';
inputn_test=mapminmax('apply',input_test,inputps);
output_test=output(n(272),:)';
for ii=1:1
        for j=1:1:midnum
            I(j)=inputn_test(:)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore=w2'*Iout'+b2;
end

output_fore=find(fore==max(fore))
correct_lable=find(output_test==1)
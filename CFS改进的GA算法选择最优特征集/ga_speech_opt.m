function[n]=ga_speech_opt(~)
%% ����
% angry=importdata('E:\aa0011\����+�Ե�\database\Angry\datasum.mat');
% clam=importdata('E:\aa0011\����+�Ե�\database\Clam\datasum_clam.mat');
% joy=importdata('E:\aa0011\����+�Ե�\database\Joy\datasum_joy.mat');
% sad=importdata('E:\aa0011\����+�Ե�\database\Sad\datasum_sad.mat');
% data=[angry;clam;joy(1:22,:);sad];
% lable1=ones(1,22)';
% lable=[lable1;lable1*2;lable1*3;lable1*4];
% data=[lable data];
% % data=data(:,1:31);
%% ����
happy=importdata('E:\aa0011\����+�Ե�\database\casia����������Ͽ�\��ͬ�ı�50\zhaozuoxiang\datasum_happy.mat');
angry=importdata('E:\aa0011\����+�Ե�\database\casia����������Ͽ�\��ͬ�ı�50\zhaozuoxiang\datasum_angry.mat');
surprise=importdata('E:\aa0011\����+�Ե�\database\casia����������Ͽ�\��ͬ�ı�50\zhaozuoxiang\datasum_surprise.mat');
sad=importdata('E:\aa0011\����+�Ե�\database\casia����������Ͽ�\��ͬ�ı�50\zhaozuoxiang\datasum_sad.mat');
fear=importdata('E:\aa0011\����+�Ե�\database\casia����������Ͽ�\��ͬ�ı�50\zhaozuoxiang\datasum_fear.mat');
neutral=importdata('E:\aa0011\����+�Ե�\database\casia����������Ͽ�\��ͬ�ı�50\zhaozuoxiang\datasum_neutral.mat');
% data=[happy(:,1:60);angry(:,1:60);surprise(:,1:60);sad(:,1:60);fear(:,1:60);neutral(:,1:60)];
data=[happy;angry;surprise;sad;fear;neutral];
data=data';
data(any(isnan(data),2),:)=[];
data=data';

lable1=ones(1,50)';
lable=[lable1;lable1*2;lable1*3;lable1*4;lable1*5;lable1*6];
data=[lable data];
data(isnan(data))=0;
%%
global  P_train T_train P_test T_test mint maxt 
global S s1 data1
S=size(data,2)-1;
s1=50;
data1=datamap(data(:,2:end));
%%
a=randperm(300);
Train=data(a(1:270),:);
Test=data(a(271:end),:);
% ѵ������
P_train=Train(:,2:end)';
T_train=Train(:,1)';
% ��������
P_test=Test(:,2:end)';
T_test=Test(:,1)';
%%
[P_train,minp,maxp,T_train,mint,maxt]=premnmx(P_train,T_train);
P_test=tramnmx(P_test,minp,maxp);
%% ��ʼ��Ⱥ
popu=50;  
bounds=ones(S,1)*[0,1];
% ������ʼ��Ⱥ
% initPop=crtbp(popu,S);
initPop=randint(popu,S,[0 1]);
% �����ʼ��Ⱥ��Ӧ��
initFit=zeros(popu,1);
for i=1:size(initPop,1)
    initFit(i)=de_code(initPop(i,:));
end
initPop=[initPop initFit];
gen=200; 
%%
[X,EndPop,BPop,Trace]=ga(bounds,'fitness',[],initPop,[1e-6 1 0],'maxGenTerm',...
    gen,'normGeomSelect',0.1,'simpleXover',8,'boundaryMutation',[2 gen 3]);
[m,n]=find(X==1);
disp(['�Ż�ɸѡ��������Ա������Ϊ:' num2str(n)]);
figure
plot(Trace(:,1),Trace(:,3),'r:')
hold on
plot(Trace(:,1),Trace(:,2),'b')
xlabel('��������')
ylabel('��Ӧ�Ⱥ���')
title('��Ӧ�Ⱥ�����������')
legend('ƽ����Ӧ�Ⱥ���','�����Ӧ�Ⱥ���')
xlim([1 gen])

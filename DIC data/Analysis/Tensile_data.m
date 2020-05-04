clear;
clc;

% importing all 0-1 data
[~, ~, raw_01] = xlsread('H:\Surfalex_All data\DIC data\Data\test 0-1\voltage data\data_1.csv','data_1','A3:P663');
data_01 = reshape([raw_01{:}],size(raw_01));

% importing all 0-2 data
[~, ~, raw_02] = xlsread('H:\Surfalex_All data\DIC data\Data\test 0-2\voltage data\data_1.csv','data_1','A3:P680');
data_02 = reshape([raw_02{:}],size(raw_02));

% importing all 30-1 data
[~, ~, raw_301] = xlsread('H:\Surfalex_All data\DIC data\Data\test 30-1\voltage data\data_1.csv','data_1','A3:P650');
data_301 = reshape([raw_301{:}],size(raw_301));

% importing all 30-2 data
[~, ~, raw_302] = xlsread('H:\Surfalex_All data\DIC data\Data\test 30-2\voltage data\data_1.csv','data_1','A3:P658');
data_302 = reshape([raw_302{:}],size(raw_302));

% importing all 45-1 data
[~, ~, raw_451] = xlsread('H:\Surfalex_All data\DIC data\Data\test 45-1\voltage data\data_1.csv','data_1','A3:P680');
data_451 = reshape([raw_451{:}],size(raw_451));

% importing all 45-2 data
[~, ~, raw_452] = xlsread('H:\Surfalex_All data\DIC data\Data\test 45-2\voltage data\data_1.csv','data_1','A3:P651');
data_452 = reshape([raw_452{:}],size(raw_452));

% importing all 60-1 data
[~, ~, raw_601] = xlsread('H:\Surfalex_All data\DIC data\Data\test 60-1\Volatge data\data_1.csv','data_1','A3:P705');
data_601 = reshape([raw_601{:}],size(raw_601));

% importing all 60-2 data
[~, ~, raw_602] = xlsread('H:\Surfalex_All data\DIC data\Data\test 60-2\voltage data\data_1.csv','data_1','A3:P604');
data_602 = reshape([raw_602{:}],size(raw_602));

% importing all 90-1 data
[~, ~, raw_901] = xlsread('H:\Surfalex_All data\DIC data\Data\test 90-1\voltage data\data_1.csv','data_1','A3:P682');
data_901 = reshape([raw_901{:}],size(raw_901));

% importing all 90-2 data
[~, ~, raw_902] = xlsread('H:\Surfalex_All data\DIC data\Data\test 90-2\voltge data\data_1.csv','data_1','A3:P645');
data_902 = reshape([raw_902{:}],size(raw_902));


%plot the true stress versus true strain curves
plot(data_01(:,5),data_01(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_02(:,5),data_02(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_301(:,5),data_301(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_302(:,5),data_302(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_451(:,5),data_451(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_452(:,5),data_452(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_601(:,5),data_601(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_602(:,5),data_602(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_901(:,5),data_901(:,16),'linestyle','-','linewidth',3);
hold on
plot(data_902(:,5),data_902(:,16),'linestyle','-','linewidth',3);

axis([0 0.35 0 350])
set(gca,'FontSize',30,'fontweight','bold')
set(gcf,'color','w');
set(gca,'linewidth',3)
xlabel('True Strain)','fontweight','bold','fontsize',32)
ylabel('True Stress (MPa)','fontweight','bold','fontsize',32)
box on
legend('0-1','0-2','30-1', '30-2','45-1','45-2','60-1','60-2','90-1','90-2','Location','EastOutside','Orientation','vertical','bold','fontsize',20);
title('True Stress vs True Strain')





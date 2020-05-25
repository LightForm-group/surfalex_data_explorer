clear;
clc;

cooktemp=[2,4,6, 10, 12, 14, 18, 20, 22]; 
cooksteps=9;
cookstepcount=1;


% importing all 10 mm data
raw_10mm = xlsread('strain path_first derivative.xlsx','Sheet1','A4:W147');


% importing all 20 mm data
raw_20mm = xlsread('strain path_first derivative.xlsx','Sheet2','A4:W142');


% importing all 40 mm data
raw_40mm = xlsread('strain path_first derivative.xlsx','Sheet3','A4:W153');


% importing all 60 mm data
raw_60mm = xlsread('strain path_first derivative.xlsx','Sheet4','A4:W157');


% importing all 120 mm data
raw_120mm = xlsread('strain path_first derivative.xlsx','Sheet5','A4:W152');


% importing all 177 mm data
raw_177mm = xlsread('strain path_first derivative.xlsx','Sheet6','A4:W159');



while cookstepcount<=cooksteps 
    scatter(raw_10mm(:,cooktemp(cookstepcount)),raw_10mm(:,cooktemp(cookstepcount)+1))
    hold on
    cookstepcount=cookstepcount+1;
    
end

cookstepcount=1;

hold on
while cookstepcount<=cooksteps 
    scatter(raw_20mm(:,cooktemp(cookstepcount)),raw_20mm(:,cooktemp(cookstepcount)+1))
    hold on
    cookstepcount=cookstepcount+1;
    
end

cookstepcount=1;

hold on
while cookstepcount<=cooksteps 
    scatter(raw_40mm(:,cooktemp(cookstepcount)),raw_40mm(:,cooktemp(cookstepcount)+1))
    hold on
    cookstepcount=cookstepcount+1;
    
end
cookstepcount=1;

 hold on
 
while cookstepcount<=cooksteps 
    scatter(raw_60mm(:,cooktemp(cookstepcount)),raw_60mm(:,cooktemp(cookstepcount)+1))
    hold on
    cookstepcount=cookstepcount+1;
    
end
cookstepcount=1;

 hold on
while cookstepcount<=cooksteps 
    scatter(raw_120mm(:,cooktemp(cookstepcount)),raw_120mm(:,cooktemp(cookstepcount)+1))
    hold on
    cookstepcount=cookstepcount+1;
    
end
cookstepcount=1;

 hold on
while cookstepcount<=cooksteps 
    scatter(raw_177mm(:,cooktemp(cookstepcount)),raw_177mm(:,cooktemp(cookstepcount)+1))
    hold on
    cookstepcount=cookstepcount+1;
    
end

set(gca,'FontSize',30,'fontweight','bold')
set(gcf,'color','w');
set(gca,'linewidth',3)
xlabel('Minor strain)','fontweight','bold','fontsize',32)
ylabel('Major strain','fontweight','bold','fontsize',32)
axis([-0.2 0.2 0 0.3])
box on
plot([0 0],get(gca,'YLim'),'k','linewidth',3);
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %Maximize figure.
print(gcf,'FLC_1st_derivative.svg','-dsvg','-r600');



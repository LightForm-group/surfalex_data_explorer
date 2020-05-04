clear;
clc;

cooktemp=[1,3,5, 8, 10, 12, 15, 17, 19]; 
cooksteps=9;
cookstepcount=1;


% importing all 10 mm data
raw_10mm = xlsread('Strain path data.xlsx','Sheet1','B4:U147');


% importing all 20 mm data
raw_20mm = xlsread('Strain path data.xlsx','Sheet2','B4:U142');


% importing all 40 mm data
raw_40mm = xlsread('Strain path data.xlsx','Sheet3','B4:U153');


% importing all 60 mm data
raw_60mm = xlsread('Strain path data.xlsx','Sheet4','B4:U157');


% importing all 120 mm data
raw_120mm = xlsread('Strain path data.xlsx','Sheet5','B4:U152');


% importing all 177 mm data
raw_177mm = xlsread('Strain path data.xlsx','Sheet6','B4:U159');



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
axis([-0.2 0.2 0 0.5])
box on
plot([0 0],get(gca,'YLim'),'k','linewidth',3);
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %Maximize figure.
print(gcf,'FLC_crack.svg','-dsvg','-r600');


